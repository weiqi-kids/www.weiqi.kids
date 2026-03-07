const { App } = require('@slack/bolt');
const { spawn } = require('child_process');
const path = require('path');
const lockfile = require('proper-lockfile');

const ALLOWED_CHANNEL = process.env.SLACK_CHANNEL_ID;
const PROJECT_PATH = process.env.PROJECT_PATH || path.join(__dirname, '..');
const ALLOWED_USERS = process.env.ALLOWED_USERS?.split(',') || [];

const app = new App({
  token: process.env.SLACK_BOT_TOKEN,
  socketMode: true,
  appToken: process.env.SLACK_APP_TOKEN,
});

app.message(async ({ message, say }) => {
  // 1. 頻道檢查
  if (message.channel !== ALLOWED_CHANNEL) return;
  if (message.subtype === 'bot_message') return;

  // 2. 使用者權限檢查
  if (ALLOWED_USERS.length > 0 && !ALLOWED_USERS.includes(message.user)) {
    await say('⛔ 您沒有權限使用此 Bot');
    return;
  }

  // 3. 輸入驗證
  try {
    validateInput(message.text);
  } catch (err) {
    await say(`⚠️ ${err.message}`);
    return;
  }

  await say('⏳ Claude 執行中，請稍候...');

  try {
    // 4. 使用檔案鎖避免並發
    const result = await executeWithLock(async () => {
      return await runClaude(message.text);
    });

    const chunks = splitMessage(result, 3000);
    for (const chunk of chunks) {
      await say(`\`\`\`\n${chunk}\n\`\`\``);
    }
    await say('✅ 執行完成！');
  } catch (err) {
    await say(`❌ 執行失敗：\n\`\`\`\n${err.message.slice(0, 1000)}\n\`\`\``);
  }
});

function validateInput(message) {
  if (message.length > 5000) {
    throw new Error('訊息過長（上限 5000 字）');
  }
  const dangerousPatterns = [/rm\s+-rf/i, /sudo\s+/i, />\s*\/etc/i];
  for (const pattern of dangerousPatterns) {
    if (pattern.test(message)) {
      throw new Error('偵測到危險指令');
    }
  }
}

async function executeWithLock(callback) {
  const release = await lockfile.lock(PROJECT_PATH, {
    retries: { retries: 5, minTimeout: 2000 }
  });
  try {
    return await callback();
  } finally {
    await release();
  }
}

function runClaude(prompt) {
  return new Promise((resolve, reject) => {
    // 注意：prompt 必須通過 stdin 傳入，不能作為參數
    const args = [
      '-p',
      '--allowedTools', 'Bash(git:*) Bash(pnpm:*) Edit Read Glob Grep Write',
      '--disallowedTools', 'Bash(rm:-rf:*) Bash(sudo:*) Bash(chmod:*)',
    ];

    const child = spawn('claude', args, {
      cwd: PROJECT_PATH,
      shell: false,  // 重要：不使用 shell
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => { stdout += data; });
    child.stderr.on('data', (data) => { stderr += data; });

    // 通過 stdin 傳入 prompt
    child.stdin.write(prompt);
    child.stdin.end();

    // 設定 timeout
    const timeout = setTimeout(() => {
      child.kill();
      reject(new Error('執行超時（10 分鐘）'));
    }, 600000);

    child.on('close', (code) => {
      clearTimeout(timeout);
      if (code === 0) {
        resolve(stdout || '（無輸出）');
      } else {
        reject(new Error(stderr || `Exit code: ${code}`));
      }
    });

    child.on('error', (err) => {
      clearTimeout(timeout);
      reject(err);
    });
  });
}

function splitMessage(text, maxLength) {
  const chunks = [];
  for (let i = 0; i < text.length; i += maxLength) {
    chunks.push(text.slice(i, i + maxLength));
  }
  return chunks;
}

(async () => {
  await app.start();
  console.log('⚡ Weiqi.Kids Slack Bot 已啟動');
  console.log(`📁 專案目錄: ${PROJECT_PATH}`);
  console.log(`👤 允許的使用者: ${ALLOWED_USERS.length > 0 ? ALLOWED_USERS.join(', ') : '所有人'}`);
})();
