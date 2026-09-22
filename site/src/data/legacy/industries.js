// 議題化分類 industry taxonomy
// 跨越 research/apps/intel 三大 .js，提供統一的產業視角給使用者篩選。
// 每個專案有一個 industry_tag，對應到下面的 id。

export const industries = [
  {
    id: 'business-intel',
    icon: '📊',
    titleKey: 'industry.business-intel.title',
    titleDefault: '商業情報',
    descKey: 'industry.business-intel.desc',
    descDefault: '產業供應鏈、貿易資訊、市場聲量、消費市場分析。',
  },
  {
    id: 'ai-math',
    icon: '🧠',
    titleKey: 'industry.ai-math.title',
    titleDefault: 'AI 理論與數學',
    descKey: 'industry.ai-math.desc',
    descDefault: '機器學習、最佳化、AI 安全、數學定理的第一性原理推導。',
  },
  {
    id: 'health',
    icon: '🩺',
    titleKey: 'industry.health.title',
    titleDefault: '醫療健康',
    descKey: 'industry.health.desc',
    descDefault: '中西醫、預防醫學、保健食品、淨水、疫情等健康面向的資訊整合。',
  },
  {
    id: 'finance-law',
    icon: '⚖️',
    titleKey: 'industry.finance-law.title',
    titleDefault: '財稅法規',
    descKey: 'industry.finance-law.desc',
    descDefault: '稅法、政策法規變動、財稅顧問、政策承諾追蹤。',
  },
  {
    id: 'education',
    icon: '📚',
    titleKey: 'industry.education.title',
    titleDefault: '教育學習',
    descKey: 'industry.education.desc',
    descDefault: '學習地圖、職涯規劃、技能教育、知識傳承。',
  },
  {
    id: 'go',
    icon: '⚫',
    titleKey: 'industry.go.title',
    titleDefault: '圍棋研究',
    descKey: 'industry.go.desc',
    descDefault: '圍棋理論、官子計算、AlphaGo 演進整理。',
  },
  {
    id: 'security',
    icon: '🛡️',
    titleKey: 'industry.security.title',
    titleDefault: '資安',
    descKey: 'industry.security.desc',
    descDefault: '資安威脅情報、漏洞追蹤、產業資安監測。',
  },
];

// industry id → display info lookup
export const industriesById = Object.fromEntries(industries.map((i) => [i.id, i]));
