"""
轻量级预测模块 - 使用 sklearn 替代 PyTorch/Kronos
适合 Streamlit Cloud 等内存受限环境
"""
import json
import os
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from ass1_core import load_bundle as _load_bundle, daily_returns as _daily_returns
    _ASS1_CORE_AVAILABLE = True
except ImportError:
    _ASS1_CORE_AVAILABLE = False
    _load_bundle = None
    _daily_returns = None


def create_features(prices: pd.Series, window: int = 20) -> pd.DataFrame:
    """从价格序列创建技术指标特征"""
    features = pd.DataFrame(index=prices.index)

    # 收益率特征
    features['returns_1d'] = prices.pct_change(1)
    features['returns_5d'] = prices.pct_change(5)
    features['returns_10d'] = prices.pct_change(10)
    features['returns_20d'] = prices.pct_change(20)

    # 移动平均线
    features['ma_5'] = prices / prices.rolling(5).mean() - 1
    features['ma_10'] = prices / prices.rolling(10).mean() - 1
    features['ma_20'] = prices / prices.rolling(20).mean() - 1
    features['ma_60'] = prices / prices.rolling(60).mean() - 1

    # 波动率
    features['volatility_10d'] = features['returns_1d'].rolling(10).std()
    features['volatility_20d'] = features['returns_1d'].rolling(20).std()

    # 价格位置 (相对于近期高低点)
    features['price_position'] = (prices - prices.rolling(20).min()) / (prices.rolling(20).max() - prices.rolling(20).min() + 1e-10)

    # 动量
    features['momentum_10d'] = prices.pct_change(10)
    features['momentum_20d'] = prices.pct_change(20)

    # 成交量特征 (如果有)

    return features.dropna()


def lightweight_forecast(
    close_df: pd.DataFrame,
    symbols: List[str] = None,
    lookback: int = 60,
    pred_len: int = 7,
    model_type: str = "ridge"
) -> Dict[str, Any]:
    """
    使用轻量级模型预测未来收益（无PyTorch依赖）

    Args:
        close_df: 收盘价 DataFrame
        symbols: 要预测的标的列表
        lookback: 历史数据长度
        pred_len: 预测未来天数
        model_type: ridge, rf

    Returns:
        与 kronos_forecast 兼容的格式
    """
    if symbols is None:
        symbols = list(close_df.columns)

    reg_results = []
    cls_results = {}

    for symbol in symbols:
        if symbol not in close_df.columns:
            continue

        try:
            prices = close_df[symbol].dropna()

            if len(prices) < lookback + 30:
                continue

            # 创建特征
            features = create_features(prices, window=20)

            if len(features) < lookback:
                continue

            # 目标：未来7天收益率
            future_returns = prices.pct_change(pred_len).shift(-pred_len)

            # 对齐数据
            aligned_data = pd.concat([features, future_returns], axis=1).dropna()
            aligned_data.columns = list(features.columns) + ['target']

            if len(aligned_data) < 30:
                continue

            X = aligned_data.iloc[:-pred_len][features.columns]
            y = aligned_data.iloc[:-pred_len]['target']

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 训练模型
            if model_type == "rf":
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            else:
                model = Ridge(alpha=1.0)

            model.fit(X_scaled, y)

            # 预测
            latest_features = features.iloc[-lookback:]
            latest_scaled = scaler.transform(latest_features)
            predictions = model.predict(latest_scaled)
            pred_cum_return = np.mean(predictions[-5:])  # 取最近几个预测的平均

            # 计算日均收益
            pred_daily_return = (1 + pred_cum_return) ** (1/pred_len) - 1

            last_close = prices.iloc[-1]
            pred_future_close = last_close * (1 + pred_cum_return)

            reg_results.append({
                "symbol": symbol,
                "pred_daily_return": float(pred_daily_return),
                "pred_7d_cum_return": float(pred_cum_return),
                "last_close": float(last_close),
                "pred_future_close": float(pred_future_close)
            })

            # 分类预测
            if pred_cum_return > 0.02:
                pred_class = 2  # Up
                probs = [0.1, 0.2, 0.7]
            elif pred_cum_return < -0.02:
                pred_class = 0  # Down
                probs = [0.7, 0.2, 0.1]
            else:
                pred_class = 1  # Flat
                probs = [0.2, 0.6, 0.2]

            cls_results[symbol] = {
                "pred_class": pred_class,
                "pred_probs": probs,
                "description": f"Lightweight {model_type} prediction",
                "pred_cum_return": float(pred_cum_return)
            }

        except Exception as e:
            print(f"Prediction failed for {symbol}: {e}")
            continue

    reg_df = pd.DataFrame(reg_results)
    if not reg_df.empty:
        reg_df = reg_df.set_index("symbol").sort_values("pred_7d_cum_return", ascending=False)

    return {
        "regression": reg_df,
        "classification": cls_results,
        "model_info": f"Lightweight {model_type} (sklearn-based, no PyTorch)"
    }


def run_lightweight_optimization(
    data_json_path: str,
    dataset: str = "universe",
    lookback: int = 60,
    pred_len: int = 7,
    model_type: str = "ridge",
    out_dir: str = None
) -> Dict[str, Any]:
    """
    运行轻量级预测并返回与现有模型兼容的格式
    """
    if not _ASS1_CORE_AVAILABLE or _load_bundle is None or _daily_returns is None:
        return {
            "regression": {},
            "classification": {},
            "model_info": "ass1_core not available",
            "metrics": {}
        }

    bundle = _load_bundle(data_json_path)

    if dataset == "assets":
        close = bundle.close_assets
    elif dataset == "stocks":
        close = bundle.close_stocks
    else:
        close = bundle.close_universe

    # Run lightweight forecast
    forecast = lightweight_forecast(
        close,
        symbols=list(close.columns),
        lookback=lookback,
        pred_len=pred_len,
        model_type=model_type
    )

    # Calculate returns for optimization
    rets = _daily_returns(close)

    result = {
        "regression": forecast["regression"].to_dict() if not forecast["regression"].empty else {},
        "classification": forecast["classification"],
        "model_info": forecast["model_info"],
        "metrics": {
            "mse": None,
            "mae": None,
            "accuracy": None
        },
        "note": "Lightweight sklearn model - no PyTorch required"
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"lightweight_{dataset}_forecast.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# 兼容 Kronos 的接口
LIGHTWEIGHT_AVAILABLE = True

def kronos_forecast(*args, **kwargs):
    """
    兼容接口：实际使用轻量级模型
    """
    # 移除 device 参数（轻量级模型不需要）
    kwargs.pop('device', None)
    kwargs.pop('model_name', None)
    return lightweight_forecast(*args, **kwargs)


if __name__ == "__main__":
    # Test
    test_data_path = "/Users/dyl/Downloads/AIE1902-Ass1-main/data.json"
    if os.path.exists(test_data_path):
        result = run_lightweight_optimization(test_data_path, dataset="stocks")
        print("Lightweight forecast completed!")
        print(f"Symbols predicted: {list(result['classification'].keys())[:5]}...")
        print(f"Model: {result['model_info']}")
    else:
        print("Test data not found")
