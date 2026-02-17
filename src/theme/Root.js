import React from 'react';
import GlobalSchema from '@site/src/components/SEO/GlobalSchema';

// 自訂 Root 組件 - 注入全域 SEO Schema
export default function Root({ children }) {
  return (
    <>
      <GlobalSchema />
      {children}
    </>
  );
}
