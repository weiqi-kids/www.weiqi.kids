import React from 'react';

/**
 * KeyTakeaway 組件 - 文章重點摘要
 * 放在文章開頭或結尾。樣式對齊 design-tokens.css（暖橙品牌）。
 *
 * @param {React.ReactNode} children - 重點內容
 * @param {string} [title] - 區塊標題（多語系用，預設「重點摘要」）
 */
export default function KeyTakeaway({ children, title = '重點摘要' }) {
  return (
    <div
      className="key-takeaway"
      style={{
        backgroundColor: 'var(--color-brand-light)',
        padding: 'var(--space-4) var(--space-5)',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--color-brand)',
        marginBottom: 'var(--space-4)',
      }}
    >
      <strong
        style={{
          display: 'block',
          marginBottom: 'var(--space-2)',
          color: 'var(--color-brand)',
        }}
      >
        📌 {title}
      </strong>
      {children}
    </div>
  );
}
