/**
 * D3Charts 組件庫 - 匯出所有 D3.js 視覺化組件
 *
 * 用法範例：
 * import { PolicyHeatmap, EloChart, MCTSTree } from '@site/src/components/D3Charts';
 */

// 基礎組件
export { default as GoBoard } from './GoBoard';

// 策略與價值視覺化
export { default as PolicyHeatmap } from './PolicyHeatmap';
// export { default as ValueSurface } from './ValueSurface';  // TODO

// 搜索與訓練視覺化
export { default as MCTSTree } from './MCTSTree';
export { default as EloChart } from './EloChart';
// export { default as SelfPlayLoop } from './SelfPlayLoop';  // TODO

// 神經網路視覺化
// export { default as FeaturePlanes } from './FeaturePlanes';  // TODO
// export { default as CNNLayers } from './CNNLayers';  // TODO
// export { default as ResNetBlock } from './ResNetBlock';  // TODO

// PUCT 公式互動
// export { default as PUCTFormula } from './PUCTFormula';  // TODO

// 系統架構
// export { default as DistributedArch } from './DistributedArch';  // TODO

// 導入共用樣式
import './styles.css';
