import React from 'react';
import Head from '@docusaurus/Head';

/**
 * FAQ 組件 - 常見問題
 * 自動產生 FAQPage Schema
 *
 * @param {Array} items - FAQ 項目陣列 [{question, answer}]
 */
export default function FAQ({ items }) {
  // 產生 FAQPage Schema
  const faqSchema = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    "mainEntity": items.map(item => ({
      "@type": "Question",
      "name": item.question,
      "acceptedAnswer": {
        "@type": "Answer",
        "text": item.answer
      }
    }))
  };

  return (
    <>
      <Head>
        <script type="application/ld+json">
          {JSON.stringify(faqSchema)}
        </script>
      </Head>
      <div className="faq-section" style={{ marginTop: '2rem' }}>
        <h2>常見問題</h2>
        {items.map((item, index) => (
          <details
            key={index}
            style={{
              marginBottom: '1rem',
              padding: '1rem',
              backgroundColor: 'var(--ifm-color-gray-100)',
              borderRadius: '8px',
            }}
          >
            <summary style={{
              fontWeight: 'bold',
              cursor: 'pointer',
              marginBottom: '0.5rem'
            }}>
              {item.question}
            </summary>
            <p className="faq-answer-content" style={{ marginTop: '0.5rem' }}>
              {item.answer}
            </p>
          </details>
        ))}
      </div>
    </>
  );
}
