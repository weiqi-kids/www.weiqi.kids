import React from 'react';

/**
 * KeyAnswer 組件 - 用於 SGE/AEO 優化
 * 直接回答搜尋問句，放在 H2 下方第一段
 *
 * @param {string} question - 對應的搜尋問句
 * @param {React.ReactNode} children - 答案內容（1-2句話）
 */
export default function KeyAnswer({ question, children }) {
  return (
    <p
      className="key-answer"
      data-question={question}
      style={{
        backgroundColor: 'var(--ifm-color-primary-lightest)',
        padding: '1rem',
        borderRadius: '8px',
        borderLeft: '4px solid var(--ifm-color-primary)',
        marginBottom: '1rem',
      }}
    >
      <strong>{children}</strong>
    </p>
  );
}
