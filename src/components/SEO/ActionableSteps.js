import React from 'react';

/**
 * ActionableSteps 組件 - 行動步驟清單
 * 提供具體行動指引
 *
 * @param {Array} steps - 步驟陣列
 */
export default function ActionableSteps({ steps }) {
  return (
    <ol
      className="actionable-steps"
      style={{
        backgroundColor: 'var(--ifm-color-info-lightest)',
        padding: '1.5rem 1.5rem 1.5rem 2.5rem',
        borderRadius: '8px',
        border: '1px solid var(--ifm-color-info)',
        marginBottom: '1rem',
      }}
    >
      {steps.map((step, index) => (
        <li key={index} style={{ marginBottom: '0.5rem' }}>
          {step}
        </li>
      ))}
    </ol>
  );
}
