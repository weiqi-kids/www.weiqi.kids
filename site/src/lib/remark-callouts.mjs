// 把舊站 Docusaurus 的 :::note / :::tip 等提示框轉成 <aside class="callout">。
// 其他被 remark-directive 誤判的 :name 文字（如「比例:abc」）還原成原文。
import { visit, SKIP } from 'unist-util-visit';

const KINDS = { note: '備註', tip: '提示', info: '說明', caution: '注意', warning: '警告', danger: '危險' };

function restoreText(node) {
  const children = node.children || [];
  const prefix = node.type === 'textDirective' ? ':' : '::';
  const label = children.length ? null : '';
  return [{ type: 'text', value: prefix + node.name + (children.length ? '[' : '') }, ...children, ...(children.length ? [{ type: 'text', value: ']' }] : []), ...(label === null ? [] : [])];
}

export default function remarkCallouts() {
  return (tree) => {
    visit(tree, (node, index, parent) => {
      if (node.type === 'containerDirective' && KINDS[node.name]) {
        let title = KINDS[node.name];
        const first = node.children[0];
        if (first?.data?.directiveLabel) {
          title = first.children.map((c) => c.value ?? '').join('') || title;
          node.children.shift();
        }
        node.data = { hName: 'aside', hProperties: { className: ['callout', `callout-${node.name}`] } };
        node.children.unshift({ type: 'paragraph', data: { hName: 'div', hProperties: { className: ['callout-title'] } }, children: [{ type: 'text', value: title }] });
        return;
      }
      if ((node.type === 'textDirective' || node.type === 'leafDirective') && parent && index != null) {
        const replacement = restoreText(node);
        if (node.type === 'leafDirective') parent.children.splice(index, 1, { type: 'paragraph', children: replacement });
        else parent.children.splice(index, 1, ...replacement);
        return [SKIP, index + replacement.length];
      }
      if (node.type === 'containerDirective' && parent && index != null) {
        parent.children.splice(index, 1, { type: 'paragraph', children: [{ type: 'text', value: ':::' + node.name }] }, ...node.children);
        return [SKIP, index];
      }
    });
  };
}
