# 🚀 Streamlit Cloud 轻量级部署版本

## 方案 B 说明

此版本使用 **轻量级 sklearn 模型** 替代 PyTorch/Kronos，专为 Streamlit Cloud 等内存受限环境设计。

## 与完整版的区别

| 特性 | 轻量级版 (ceshi) | 完整版 (母文件夹) |
|------|------------------|-------------------|
| 预测模型 | sklearn (Ridge/RF) | PyTorch Kronos |
| 内存占用 | ~100MB | ~600MB+ |
| 模型精度 | 中等 | 较高 |
| 安装速度 | 快（2分钟） | 慢（10分钟+） |
| 网络依赖 | 无需下载模型 | 需下载或包含模型 |

## 部署步骤

### 1. 上传代码到 GitHub

```bash
cd "/Users/dyl/Downloads/AIE1902-Ass1-main 2/ceshi"
git init
git add .
git commit -m "Add lightweight cloud version"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### 2. Streamlit Cloud 部署

1. 访问 https://streamlit.io/cloud
2. 登录并点击 **New app**
3. 选择你的仓库和分支
4. 主文件路径：`app.py`
5. 点击 **Deploy**

## 文件说明

- `lightweight_predictor.py` - 轻量级预测模型（sklearn）
- `kronos_predictor.py` - 适配器，自动选择模型
- `app.py` - 主应用（兼容两种模式）
- `requirements.txt` - 依赖（无 torch）

## 本地测试

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 故障排除

**问题**: lightgbm 安装失败
**解决**: 在 packages.txt 中取消 libgomp1 的注释

**问题**: 预测结果与本地版不同
**解决**: 这是正常的，轻量级模型使用不同的算法

## 技术细节

轻量级模型使用以下特征：
- 历史收益率 (1d, 5d, 10d, 20d)
- 移动平均线相对位置
- 波动率指标
- 价格动量

预测使用 Ridge 回归或 Random Forest，在云端 1GB 内存限制下稳定运行。
