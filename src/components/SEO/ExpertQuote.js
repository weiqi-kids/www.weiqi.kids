import React from 'react';

/**
 * ExpertQuote 組件 - 專家引言
 * 增加 E-E-A-T 權威性
 *
 * @param {string} author - 專家姓名
 * @param {string} title - 專家頭銜
 * @param {React.ReactNode} children - 引言內容
 */
export default function ExpertQuote({ author, title, children }) {
  return (
    <blockquote
      className="expert-quote"
      style={{
        backgroundColor: 'var(--ifm-color-gray-100)',
        padding: '1.5rem',
        borderRadius: '8px',
        borderLeft: '4px solid var(--ifm-color-primary)',
        fontStyle: 'italic',
        margin: '1.5rem 0',
      }}
    >
      <p style={{ marginBottom: '0.5rem' }}>{children}</p>
      <cite style={{ fontStyle: 'normal', fontSize: '0.9rem' }}>
        — {author}{title && `, ${title}`}
      </cite>
    </blockquote>
  );
}
