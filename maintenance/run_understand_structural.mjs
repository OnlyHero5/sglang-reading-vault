#!/usr/bin/env node
/**
 * Deterministic structural /understand pipeline for large repos when LLM
 * batch analysis is not run inline. Uses extract-structure.mjs per batch.
 */
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = process.argv[2] || resolve(__dirname, '../slime');
const SKILL_DIR = resolve(process.env.UA_SKILL_DIR || 'C:/Users/psx/.cursor/skills/understand');
const INTER = join(PROJECT_ROOT, '.understand-anything/intermediate');
const TMP = join(PROJECT_ROOT, '.understand-anything/tmp');
const EXTRACT = join(SKILL_DIR, 'extract-structure.mjs');
const MERGE_PY = join(SKILL_DIR, 'merge-batch-graphs.py');
const FINGERPRINT = join(SKILL_DIR, 'build-fingerprints.mjs');

mkdirSync(TMP, { recursive: true });
mkdirSync(INTER, { recursive: true });

const configPath = join(PROJECT_ROOT, '.understand-anything/config.json');
writeFileSync(configPath, JSON.stringify({ outputLanguage: 'zh', autoUpdate: false }, null, 2));

const batchesData = JSON.parse(readFileSync(join(INTER, 'batches.json'), 'utf8'));
const scanData = JSON.parse(readFileSync(join(INTER, 'scan-result.json'), 'utf8'));
const commit = spawnSync('git', ['-C', PROJECT_ROOT, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).stdout.trim();

function nodeTypeFor(file) {
  const { fileCategory, path, language } = file;
  if (fileCategory === 'config') return 'config';
  if (fileCategory === 'docs') return 'document';
  if (fileCategory === 'data') {
    if (path.endsWith('.sql')) return 'table';
    if (path.endsWith('.proto') || path.endsWith('.graphql')) return 'schema';
    return 'schema';
  }
  if (fileCategory === 'infra') {
    if (/workflow|gitlab-ci|jenkins|circleci/i.test(path)) return 'pipeline';
    if (/\.tf$|cloudformation|vagrant/i.test(path)) return 'resource';
    return 'service';
  }
  return 'file';
}

function prefixForType(type) {
  const map = {
    file: 'file', config: 'config', document: 'document', service: 'service',
    pipeline: 'pipeline', resource: 'resource', table: 'table', schema: 'schema', endpoint: 'endpoint',
  };
  return map[type] || 'file';
}

function zhSummary(file, result) {
  const lines = result?.nonEmptyLines ?? file.sizeLines ?? 0;
  const fc = result?.functions?.length ?? 0;
  const cc = result?.classes?.length ?? 0;
  const cat = file.fileCategory;
  const lang = file.language || 'unknown';
  if (cat === 'docs') return `${file.path}：${lang} 文档，约 ${lines} 行。`;
  if (cat === 'config') return `${file.path}：项目配置文件（${lang}）。`;
  if (cat === 'infra') return `${file.path}：基础设施/部署相关文件。`;
  if (fc || cc) return `${file.path}：${lang} 源文件，约 ${lines} 行，含 ${fc} 个函数、${cc} 个类。`;
  return `${file.path}：${lang} 源文件，约 ${lines} 行。`;
}

function complexity(result) {
  const lines = result?.nonEmptyLines ?? 0;
  if (lines > 200) return 'complex';
  if (lines > 50) return 'moderate';
  return 'simple';
}

function tagsFor(file, result) {
  const tags = [file.language || 'unknown', file.fileCategory];
  const base = file.path.split('/').pop() || file.path;
  if (/test|spec|_test\./i.test(file.path)) tags.push('test');
  if (base === 'train.py' || base === 'train_async.py') tags.push('entry-point');
  if (file.path.startsWith('slime/rollout')) tags.push('rollout');
  if (file.path.startsWith('slime/backends')) tags.push('training');
  if ((result?.functions?.length ?? 0) > 5) tags.push('utility');
  return [...new Set(tags)].slice(0, 5);
}

function buildBatchGraph(batch) {
  const { batchIndex, files, batchImportData = {} } = batch;
  const inputPath = join(TMP, `ua-input-${batchIndex}.json`);
  const extractPath = join(TMP, `ua-extract-${batchIndex}.json`);
  writeFileSync(inputPath, JSON.stringify({
    projectRoot: PROJECT_ROOT,
    batchFiles: files,
    batchImportData,
  }, null, 2));

  const r = spawnSync(process.execPath, [EXTRACT, inputPath, extractPath], {
    encoding: 'utf8',
    maxBuffer: 50 * 1024 * 1024,
  });
  if (r.status !== 0) {
    throw new Error(`extract-structure batch ${batchIndex}: ${r.stderr || r.stdout}`);
  }
  const extracted = JSON.parse(readFileSync(extractPath, 'utf8'));
  const nodes = [];
  const edges = [];
  const resultByPath = new Map((extracted.results || []).map(x => [x.path, x]));

  for (const file of files) {
    const result = resultByPath.get(file.path) || {};
    const type = nodeTypeFor(file);
    const prefix = prefixForType(type);
    const nodeId = `${prefix}:${file.path}`;
    nodes.push({
      id: nodeId,
      type,
      name: file.path.split('/').pop() || file.path,
      filePath: file.path,
      summary: zhSummary(file, result),
      tags: tagsFor(file, result),
      complexity: complexity(result),
      language: file.language,
    });

    for (const fn of result.functions || []) {
      const lineSpan = (fn.endLine ?? fn.startLine) - (fn.startLine ?? 0);
      if (lineSpan < 10 && !fn.exported) continue;
      const fnId = `function:${file.path}:${fn.name}`;
      nodes.push({
        id: fnId,
        type: 'function',
        name: fn.name,
        filePath: file.path,
        summary: `${fn.name}：定义于 ${file.path} 第 ${fn.startLine} 行附近。`,
        tags: ['function', file.language || 'code'],
        complexity: lineSpan > 50 ? 'moderate' : 'simple',
      });
      edges.push({ source: nodeId, target: fnId, type: 'contains', weight: 1.0 });
    }

    for (const cls of result.classes || []) {
      const lineSpan = (cls.endLine ?? cls.startLine) - (cls.startLine ?? 0);
      if (lineSpan < 20 && (cls.methods?.length ?? 0) < 2) continue;
      const clsId = `class:${file.path}:${cls.name}`;
      nodes.push({
        id: clsId,
        type: 'class',
        name: cls.name,
        filePath: file.path,
        summary: `${cls.name}：类定义于 ${file.path}。`,
        tags: ['class', file.language || 'code'],
        complexity: lineSpan > 100 ? 'complex' : 'moderate',
      });
      edges.push({ source: nodeId, target: clsId, type: 'contains', weight: 1.0 });
    }

    for (const imp of batchImportData[file.path] || []) {
      const impType = imp.endsWith('.md') ? 'document' : 'file';
      const impPrefix = prefixForType(impType);
      edges.push({
        source: nodeId,
        target: `${impPrefix}:${imp}`,
        type: 'imports',
        weight: 0.7,
      });
    }
  }

  writeFileSync(join(INTER, `batch-${batchIndex}.json`), JSON.stringify({ nodes, edges }, null, 2));
  process.stderr.write(`batch ${batchIndex}: ${nodes.length} nodes, ${edges.length} edges\n`);
}

for (const batch of batchesData.batches) {
  buildBatchGraph(batch);
}

const merge = spawnSync('python', [MERGE_PY, PROJECT_ROOT], { encoding: 'utf8', maxBuffer: 50 * 1024 * 1024 });
if (merge.status !== 0) throw new Error(`merge failed: ${merge.stderr}`);
process.stderr.write(merge.stderr || '');

const assembled = JSON.parse(readFileSync(join(INTER, 'assembled-graph.json'), 'utf8'));
const fileNodeIds = assembled.nodes
  .filter(n => ['file', 'config', 'document', 'service', 'pipeline', 'resource'].includes(n.type))
  .map(n => n.id);

function layer(id, name, desc, match) {
  return {
    id: `layer:${id}`,
    name,
    description: desc,
    nodeIds: fileNodeIds.filter(fid => {
      const p = fid.split(':').slice(1).join(':');
      return match(p);
    }),
  };
}

const assigned = new Set();
const finalLayers = [];
for (const l of [
  layer('core', '核心运行时', 'slime 包内训练、Rollout、Ray 编排与后端。', p => p.startsWith('slime/')),
  layer('examples', '示例与脚本', 'examples 目录下的 RL 训练示例。', p => p.startsWith('examples/')),
  layer('tools', '工具链', '权重转换、数据处理等 tools 脚本。', p => p.startsWith('tools/')),
  layer('docs', '文档', 'Sphinx 与项目文档。', p => p.startsWith('docs/')),
  layer('docker', '容器与补丁', 'Docker 镜像与 vendor 补丁。', p => p.startsWith('docker/')),
  layer('tests', '测试', '单元与集成测试。', p => p.startsWith('tests/')),
]) {
  l.nodeIds = l.nodeIds.filter(id => !assigned.has(id));
  l.nodeIds.forEach(id => assigned.add(id));
  if (l.nodeIds.length) finalLayers.push(l);
}
finalLayers.push({
  id: 'layer:misc',
  name: '其他',
  description: '未归入上述层的文件。',
  nodeIds: fileNodeIds.filter(id => !assigned.has(id)),
});

const pick = (pred) => assembled.nodes.find(n => pred(n))?.id;
const tour = [
  { order: 1, title: '项目概览', description: '从 README 了解 Slime 的 RL 后训练定位。', nodeIds: [pick(n => n.filePath === 'README.md')].filter(Boolean) },
  { order: 2, title: '训练入口', description: 'train.py / train_async.py 启动 Ray + Megatron + Rollout 闭环。', nodeIds: ['file:train.py', 'file:train_async.py'].filter(id => fileNodeIds.includes(id)) },
  { order: 3, title: 'Rollout 生成', description: 'RolloutManager 与 SGLang 引擎拓扑。', nodeIds: fileNodeIds.filter(id => id.includes('rollout')).slice(0, 5) },
  { order: 4, title: 'Megatron 训练后端', description: 'Actor 初始化、训练步与 loss 计算。', nodeIds: fileNodeIds.filter(id => id.includes('backends/megatron')).slice(0, 5) },
  { order: 5, title: '权重同步', description: '训练权重推送到 Rollout 引擎。', nodeIds: fileNodeIds.filter(id => id.includes('update_weight') || id.includes('weight')).slice(0, 5) },
].filter(s => s.nodeIds.length);

const graph = {
  version: '1.0.0',
  project: {
    name: 'Slime',
    languages: scanData.stats?.byLanguage ? Object.keys(scanData.stats.byLanguage) : ['python'],
    frameworks: ['Ray', 'Megatron', 'SGLang'],
    description: '面向 LLM RL 后训练的 Megatron 训练 + SGLang Rollout 框架（结构图谱，中文摘要）。',
    analyzedAt: new Date().toISOString(),
    gitCommitHash: commit,
  },
  nodes: assembled.nodes,
  edges: assembled.edges,
  layers: finalLayers,
  tour,
};

writeFileSync(join(PROJECT_ROOT, '.understand-anything/knowledge-graph.json'), JSON.stringify(graph, null, 2));

const sourcePaths = (scanData.files || []).map(f => f.path);
writeFileSync(join(INTER, 'fingerprint-input.json'), JSON.stringify({
  projectRoot: PROJECT_ROOT,
  sourceFilePaths: sourcePaths,
  gitCommitHash: commit,
}, null, 2));

const fp = spawnSync(process.execPath, [FINGERPRINT, join(INTER, 'fingerprint-input.json')], { encoding: 'utf8' });
if (fp.status !== 0 || !`${fp.stdout}${fp.stderr}`.includes('Fingerprints baseline:')) {
  throw new Error(`fingerprints failed: ${fp.stderr || fp.stdout}`);
}

writeFileSync(join(PROJECT_ROOT, '.understand-anything/meta.json'), JSON.stringify({
  lastAnalyzedAt: graph.project.analyzedAt,
  gitCommitHash: commit,
  version: '1.0.0',
  analyzedFiles: sourcePaths.length,
}, null, 2));

console.log(JSON.stringify({
  ok: true,
  path: join(PROJECT_ROOT, '.understand-anything/knowledge-graph.json'),
  nodes: graph.nodes.length,
  edges: graph.edges.length,
  layers: graph.layers.length,
  tourSteps: graph.tour.length,
  commit,
}, null, 2));
