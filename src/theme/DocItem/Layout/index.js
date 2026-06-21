import React from 'react';
import Layout from '@theme-original/DocItem/Layout';
import DocSchema from '@site/src/components/SEO/DocSchema';

/**
 * 包裝 DocItem/Layout：在每個 docs 頁面自動注入頁面層 JSON-LD（DocSchema）。
 * 不改變原本版面，只在 <Head> 注入結構化資料。
 */
export default function LayoutWrapper(props) {
  return (
    <>
      <DocSchema />
      <Layout {...props} />
    </>
  );
}
