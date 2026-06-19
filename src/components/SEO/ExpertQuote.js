import React from 'react';

/**
 * ExpertQuote 組件 - 專家引言
 * 增加 E-E-A-T 權威性。樣式對齊 design-tokens.css（暖橙品牌）。
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
        backgroundColor: 'var(--color-brand-light)',
        padding: 'var(--space-5)',
        borderRadius: 'var(--radius-md)',
        borderLeft: '4px solid var(--color-brand)',
        fontStyle: 'italic',
        color: 'var(--color-text-primary)',
        margin: 'var(--space-6) 0',
      }}
    >
      {/* 用 div 包 children，避免 MDX 段落造成 <p> 巢狀的空白段落 */}
      <div style={{ marginBottom: 'var(--space-2)' }}>{children}</div>
      <cite
        style={{
          fontStyle: 'normal',
          fontSize: 'var(--font-size-caption)',
          color: 'var(--color-text-muted)',
        }}
      >
        — {author}{title && `, ${title}`}
      </cite>
    </blockquote>
  );
}
