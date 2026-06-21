import React from 'react';
import Img from '@theme-original/MDXComponents/Img';

/**
 * 包裝 MDX 的 <img>：預設加上 lazy loading 與 async decoding，
 * 改善所有文件內圖片的載入效能（個別圖片仍可自行指定 loading 覆蓋）。
 */
export default function ImgWrapper(props) {
  return <Img loading="lazy" decoding="async" {...props} />;
}
