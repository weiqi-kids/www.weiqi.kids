import React from 'react';

/**
 * KeyTakeaway 組件 - 文章重點摘要
 * 放在文章開頭或結尾
 *
 * @param {React.ReactNode} children - 重點內容
 */
export default function KeyTakeaway({ children }) {
  return (
    <div
      className="key-takeaway"
      style={{
        backgroundColor: 'var(--ifm-color-success-lightest)',
        padding: '1rem',
        borderRadius: '8px',
        border: '1px solid var(--ifm-color-success)',
        marginBottom: '1rem',
      }}
    >
      <strong style={{ display: 'block', marginBottom: '0.5rem' }}>
        📌 重點摘要
      </strong>
      {children}
    </div>
  );
}
