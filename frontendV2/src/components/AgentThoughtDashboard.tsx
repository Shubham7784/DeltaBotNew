import React, { useState, useEffect, useCallback } from 'react';
import { apiFetch } from '../lib/api';
import {
  PaperAccountData,
  PaperPositionData,
  PaperOrderData,
  SupervisorSessionData,
  DecisionTraceData,
  AnalyticsPerformanceData,
  WorkerDriftData,
} from '../types';
import {
  Brain,
  Shield,
  Zap,
  TrendingUp,
  TrendingDown,
  Activity,
  CheckCircle2,
  XCircle,
  Clock,
  RefreshCw,
  Play,
  Layers,
  ArrowRight,
  Database,
  BarChart2,
  AlertTriangle,
  Lock,
} from 'lucide-react';

export const AgentThoughtDashboard: React.FC = () => {
  // Data States
  const [account, setAccount] = useState<PaperAccountData | null>(null);
  const [positions, setPositions] = useState<PaperPositionData[]>([]);
  const [orders, setOrders] = useState<PaperOrderData[]>([]);
  const [sessions, setSessions] = useState<SupervisorSessionData[]>([]);
  const [traces, setTraces] = useState<DecisionTraceData[]>([]);
  const [performance, setPerformance] = useState<AnalyticsPerformanceData | null>(null);
  const [drift, setDrift] = useState<WorkerDriftData | null>(null);

  // UI States
  const [activeTab, setActiveTab] = useState<'positions' | 'thoughts' | 'traces'>('thoughts');
  const [selectedAsset, setSelectedAsset] = useState<'BTC' | 'ETH'>('BTC');
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isTriggeringCycle, setIsTriggeringCycle] = useState<boolean>(false);
  const [triggerFeedback, setTriggerFeedback] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);

  // Fetch all live data from DeltaBot 2.0 REST endpoints
  const fetchAllData = useCallback(async () => {
    try {
      const [
        accRes,
        posRes,
        ordRes,
        sessRes,
        tracesRes,
        perfRes,
        driftRes,
      ] = await Promise.allSettled([
        apiFetch('/api/v1/trading/account'),
        apiFetch('/api/v1/trading/positions'),
        apiFetch('/api/v1/trading/orders'),
        apiFetch('/api/v1/supervisor/sessions'),
        apiFetch('/api/v1/supervisor/traces?limit=25'),
        apiFetch('/api/v1/analytics/performance'),
        apiFetch('/api/v1/analytics/worker-drift'),
      ]);

      if (accRes.status === 'fulfilled' && accRes.value.ok) {
        setAccount(await accRes.value.json());
      }
      if (posRes.status === 'fulfilled' && posRes.value.ok) {
        setPositions(await posRes.value.json());
      }
      if (ordRes.status === 'fulfilled' && ordRes.value.ok) {
        setOrders(await ordRes.value.json());
      }
      if (sessRes.status === 'fulfilled' && sessRes.value.ok) {
        setSessions(await sessRes.value.json());
      }
      if (tracesRes.status === 'fulfilled' && tracesRes.value.ok) {
        const traceList: DecisionTraceData[] = await tracesRes.value.json();
        setTraces(traceList);
        if (traceList.length > 0 && !selectedTraceId) {
          setSelectedTraceId(traceList[0].id);
        }
      }
      if (perfRes.status === 'fulfilled' && perfRes.value.ok) {
        setPerformance(await perfRes.value.json());
      }
      if (driftRes.status === 'fulfilled' && driftRes.value.ok) {
        setDrift(await driftRes.value.json());
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedTraceId]);

  useEffect(() => {
    fetchAllData();
    if (!autoRefresh) return;
    const interval = setInterval(fetchAllData, 4000);
    return () => clearInterval(interval);
  }, [fetchAllData, autoRefresh]);

  // Trigger manual on-demand cycle
  const handleTriggerCycle = async (asset: 'BTC' | 'ETH') => {
    setIsTriggeringCycle(true);
    setTriggerFeedback(null);
    try {
      const res = await apiFetch('/api/v1/supervisor/cycle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ asset, stage: 'monitor' }),
      });
      if (res.ok) {
        const data = await res.json();
        const decision = data?.result?.validator_decision || 'CYCLE_EXECUTED';
        setTriggerFeedback(`Successfully ran cycle for ${asset}: Decision = ${decision}`);
        await fetchAllData();
      } else {
        setTriggerFeedback(`Cycle trigger failed (${res.statusText})`);
      }
    } catch (err) {
      setTriggerFeedback(`Error triggering cycle: ${String(err)}`);
    } finally {
      setIsTriggeringCycle(false);
    }
  };

  // Find latest trace for selected asset or by selectedTraceId
  const latestTraceForAsset = traces.find(
    (t) => (t.details?.asset === selectedAsset || t.details?.technical_analysis?.asset === selectedAsset)
  ) || traces[0];

  const activeTrace = traces.find((t) => t.id === selectedTraceId) || latestTraceForAsset;

  const openPositions = positions.filter((p) => p.status === 'OPEN');
  const closedPositions = positions.filter((p) => p.status === 'CLOSED');

  return (
    <div className="space-y-6">
      {/* Top Banner: Real-time Account Summary */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 pb-5 border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
              <Brain className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold text-slate-100 tracking-tight">DeltaBot 2.0 Live Operations</h1>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                  PAPER TRADING SIMULATOR
                </span>
                {account?.is_halted && (
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold bg-rose-500/20 text-rose-300 border border-rose-500/30">
                    EMERGENCY HALT
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Deterministic Supervisor • Specialist Worker Intelligence • Risk Engine Authorized
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <button
              onClick={() => handleTriggerCycle('BTC')}
              disabled={isTriggeringCycle}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold shadow-sm transition-colors"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Step BTC</span>
            </button>
            <button
              onClick={() => handleTriggerCycle('ETH')}
              disabled={isTriggeringCycle}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-semibold shadow-sm transition-colors"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Step ETH</span>
            </button>
            <button
              onClick={() => fetchAllData()}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold transition-colors border border-slate-700"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Refresh</span>
            </button>
            <label className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 text-xs text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded border-slate-700 text-emerald-500 focus:ring-0"
              />
              <span className="font-mono text-[11px]">Auto (4s)</span>
            </label>
          </div>
        </div>

        {/* Trigger feedback toast */}
        {triggerFeedback && (
          <div className="mt-3 p-2.5 rounded-lg bg-indigo-950/60 border border-indigo-500/40 text-indigo-200 text-xs flex items-center justify-between">
            <span>{triggerFeedback}</span>
            <button onClick={() => setTriggerFeedback(null)} className="text-indigo-400 hover:text-indigo-200 ml-2">
              ✕
            </button>
          </div>
        )}

        {/* Account Metric Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 mt-5">
          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Total Equity</span>
            <div className="text-xl sm:text-2xl font-bold font-mono text-emerald-400 mt-1">
              ${account ? account.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '10,000.00'}
            </div>
            <span className="text-[10px] text-slate-500">Simulated Capital</span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Cash Balance</span>
            <div className="text-xl sm:text-2xl font-bold font-mono text-slate-200 mt-1">
              ${account ? account.cash_balance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '10,000.00'}
            </div>
            <span className="text-[10px] text-slate-500">Unencumbered</span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Daily Realized P&L</span>
            <div className={`text-xl sm:text-2xl font-bold font-mono mt-1 ${
              (account?.daily_realized_pnl || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
            }`}>
              {(account?.daily_realized_pnl || 0) >= 0 ? '+' : ''}$
              {account ? account.daily_realized_pnl.toFixed(2) : '0.00'}
            </div>
            <span className="text-[10px] text-slate-500">Resets 00:00 UTC</span>
          </div>

          <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5">
            <span className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">Open Positions</span>
            <div className="text-xl sm:text-2xl font-bold font-mono text-indigo-400 mt-1">
              {openPositions.length} <span className="text-xs font-normal text-slate-400">/ 2 max</span>
            </div>
            <span className="text-[10px] text-slate-500">BTC & ETH Independent</span>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex border-b border-slate-800 gap-2">
        <button
          onClick={() => setActiveTab('thoughts')}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 transition-all ${
            activeTab === 'thoughts'
              ? 'border-indigo-500 text-indigo-400 bg-slate-900/40 rounded-t-lg'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Brain className="w-4 h-4" />
          <span>Agent Thought Process</span>
          <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-indigo-500/20 text-indigo-300 font-mono">
            Reasoning Engine
          </span>
        </button>

        <button
          onClick={() => setActiveTab('positions')}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 transition-all ${
            activeTab === 'positions'
              ? 'border-emerald-500 text-emerald-400 bg-slate-900/40 rounded-t-lg'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Activity className="w-4 h-4" />
          <span>Running Positions & Orders</span>
          {openPositions.length > 0 && (
            <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-emerald-500/20 text-emerald-300 font-mono">
              {openPositions.length} active
            </span>
          )}
        </button>

        <button
          onClick={() => setActiveTab('traces')}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-semibold border-b-2 transition-all ${
            activeTab === 'traces'
              ? 'border-indigo-500 text-indigo-400 bg-slate-900/40 rounded-t-lg'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Database className="w-4 h-4" />
          <span>Audit Decision Traces</span>
          <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-slate-800 text-slate-300 font-mono">
            {traces.length}
          </span>
        </button>
      </div>

      {/* ========================================================================= */}
      {/* TAB 1: THOUGHT PROCESS (The Multi-Agent "Brain" Pipeline) */}
      {/* ========================================================================= */}
      {activeTab === 'thoughts' && (
        <div className="space-y-6">
          {/* Asset Selector & Regime Header */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 bg-slate-900/70 border border-slate-800 p-4 rounded-xl">
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400 font-mono">Asset Focus:</span>
              <button
                onClick={() => setSelectedAsset('BTC')}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                  selectedAsset === 'BTC'
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
              >
                BTCUSD
              </button>
              <button
                onClick={() => setSelectedAsset('ETH')}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                  selectedAsset === 'ETH'
                    ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 shadow-sm'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
              >
                ETHUSD
              </button>
            </div>

            <div className="flex items-center gap-4 text-xs font-mono">
              <div className="flex items-center gap-1.5 text-slate-300">
                <span className="text-slate-500">Market Regime:</span>
                <span className="px-2 py-0.5 rounded bg-slate-800 text-indigo-300 border border-slate-700 uppercase">
                  {activeTrace?.details?.regime || 'RANGE'}
                </span>
              </div>
              <div className="flex items-center gap-1.5 text-slate-300">
                <span className="text-slate-500">Session Bias:</span>
                <span className={`px-2 py-0.5 rounded border uppercase font-bold ${
                  activeTrace?.details?.session_bias === 'LONG_ONLY'
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                    : activeTrace?.details?.session_bias === 'SHORT_ONLY'
                    ? 'bg-rose-500/10 text-rose-400 border-rose-500/30'
                    : 'bg-slate-800 text-slate-400 border-slate-700'
                }`}>
                  {activeTrace?.details?.session_bias || 'NO_TRADE'}
                </span>
              </div>
            </div>
          </div>

          {/* Visual Step-by-Step Multi-Agent Reasoning Pipeline */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Step 1: Technical Analysis Worker */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 font-mono text-xs flex items-center justify-center font-bold">1</span>
                    <h3 className="text-xs font-bold text-slate-200">Technical Worker</h3>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">Worker #4</span>
                </div>

                <div className="mt-3 space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Setup Pattern:</span>
                    <span className="font-mono font-semibold text-indigo-300">
                      {activeTrace?.details?.technical_analysis?.setup_candidate || 'NO_SETUP'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">RSI (14):</span>
                    <span className="font-mono text-slate-200">
                      {activeTrace?.details?.technical_analysis?.rsi_14 !== undefined
                        ? Number(activeTrace.details.technical_analysis.rsi_14).toFixed(1)
                        : '50.0'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">EMA Fast/Slow:</span>
                    <span className="font-mono text-slate-300">
                      {activeTrace?.details?.technical_analysis?.ema_fast
                        ? `$${Math.round(activeTrace.details.technical_analysis.ema_fast)} / $${Math.round(activeTrace.details.technical_analysis.ema_slow)}`
                        : 'Aligned'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">VWAP Interaction:</span>
                    <span className="font-mono text-slate-300">
                      {activeTrace?.details?.technical_analysis?.vwap
                        ? `$${Number(activeTrace.details.technical_analysis.vwap).toFixed(1)}`
                        : 'Neutral'}
                    </span>
                  </div>
                </div>
              </div>
              <div className="mt-4 pt-2 border-t border-slate-800/60 text-[11px] text-slate-400">
                Pattern Recognition: {activeTrace?.details?.technical_analysis?.breakout_detected ? 'Breakout Detected' : 'No Breakout'}
              </div>
            </div>

            {/* Step 2: ML Prediction Worker */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 font-mono text-xs flex items-center justify-center font-bold">2</span>
                    <h3 className="text-xs font-bold text-slate-200">ML Prediction</h3>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">Worker #5</span>
                </div>

                <div className="mt-3 space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Direction:</span>
                    <span className={`font-mono font-bold ${
                      activeTrace?.details?.ml_prediction?.prediction === 'LONG'
                        ? 'text-emerald-400'
                        : activeTrace?.details?.ml_prediction?.prediction === 'SHORT'
                        ? 'text-rose-400'
                        : 'text-slate-400'
                    }`}>
                      {activeTrace?.details?.ml_prediction?.prediction || 'NO_TRADE'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Calibrated Conf:</span>
                    <span className="font-mono text-slate-200">
                      {activeTrace?.details?.ml_prediction?.confidence !== undefined
                        ? `${(Number(activeTrace.details.ml_prediction.confidence) * 100).toFixed(1)}%`
                        : '0.0%'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Primary Model:</span>
                    <span className="font-mono text-slate-300 truncate max-w-[120px]">
                      LightGBM (Approved)
                    </span>
                  </div>
                </div>

                {/* Probability Distribution Bar */}
                <div className="mt-3 pt-2">
                  <span className="text-[10px] text-slate-500 block mb-1">Class Probabilities:</span>
                  <div className="flex h-2 w-full rounded-full overflow-hidden bg-slate-800">
                    <div
                      style={{ width: `${(activeTrace?.details?.ml_prediction?.probabilities?.LONG || 0.33) * 100}%` }}
                      className="bg-emerald-500"
                      title="Long Prob"
                    />
                    <div
                      style={{ width: `${(activeTrace?.details?.ml_prediction?.probabilities?.NO_TRADE || 0.34) * 100}%` }}
                      className="bg-slate-600"
                      title="No Trade Prob"
                    />
                    <div
                      style={{ width: `${(activeTrace?.details?.ml_prediction?.probabilities?.SHORT || 0.33) * 100}%` }}
                      className="bg-rose-500"
                      title="Short Prob"
                    />
                  </div>
                  <div className="flex justify-between text-[9px] font-mono text-slate-400 mt-1">
                    <span>L: {((activeTrace?.details?.ml_prediction?.probabilities?.LONG || 0) * 100).toFixed(0)}%</span>
                    <span>N: {((activeTrace?.details?.ml_prediction?.probabilities?.NO_TRADE || 0) * 100).toFixed(0)}%</span>
                    <span>S: {((activeTrace?.details?.ml_prediction?.probabilities?.SHORT || 0) * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
              <div className="mt-4 pt-2 border-t border-slate-800/60 text-[11px] text-slate-400">
                Gate: Probability &gt; 45% threshold
              </div>
            </div>

            {/* Step 3: Order Flow Worker */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 font-mono text-xs flex items-center justify-center font-bold">3</span>
                    <h3 className="text-xs font-bold text-slate-200">Order Flow</h3>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">Worker #6</span>
                </div>

                <div className="mt-3 space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Imbalance:</span>
                    <span className="font-mono text-slate-200">
                      {activeTrace?.details?.order_flow?.bid_ask_imbalance !== undefined
                        ? `${(Number(activeTrace.details.order_flow.bid_ask_imbalance) * 100).toFixed(1)}%`
                        : '0.0%'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Bid/Ask Spread:</span>
                    <span className="font-mono text-slate-200">
                      {activeTrace?.details?.order_flow?.spread_bps !== undefined
                        ? `${Number(activeTrace.details.order_flow.spread_bps).toFixed(1)} bps`
                        : '1.2 bps'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Data Freshness:</span>
                    <span className="font-mono text-emerald-400">
                      {activeTrace?.details?.order_flow?.status || 'AVAILABLE'}
                    </span>
                  </div>
                </div>
              </div>
              <div className="mt-4 pt-2 border-t border-slate-800/60 text-[11px] text-slate-400">
                Orderbook Depth: Active L2 Stream
              </div>
            </div>

            {/* Step 4: Trade Validator & Gatekeeper */}
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 font-mono text-xs flex items-center justify-center font-bold">4</span>
                    <h3 className="text-xs font-bold text-slate-200">Trade Validator</h3>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">Worker #7</span>
                </div>

                <div className="mt-3 space-y-2 text-xs">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400">Final Verdict:</span>
                    <span className={`px-2 py-0.5 rounded text-[11px] font-bold font-mono ${
                      activeTrace?.details?.validator_decision === 'APPROVE'
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                        : activeTrace?.details?.validator_decision === 'REJECT'
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                        : 'bg-slate-800 text-slate-300 border border-slate-700'
                    }`}>
                      {activeTrace?.details?.validator_decision || 'NO_TRADE'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Proposal ID:</span>
                    <span className="font-mono text-slate-400 truncate max-w-[120px]">
                      {activeTrace?.details?.proposal?.id || 'None'}
                    </span>
                  </div>
                </div>

                {/* Validator Rationale Box */}
                <div className="mt-3 p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] text-slate-300">
                  <span className="font-semibold text-slate-400 block mb-0.5 font-mono">Reasoning:</span>
                  <p className="line-clamp-3">
                    {activeTrace?.details?.rationale || 'Session bias is NO_TRADE (gate locked). Preserving capital.'}
                  </p>
                </div>
              </div>
              <div className="mt-4 pt-2 border-t border-slate-800/60 text-[11px] text-slate-400">
                Independent Risk Authorization Required
              </div>
            </div>
          </div>

          {/* Full Thought Process Details Narrative */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <div className="flex items-center justify-between pb-4 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-indigo-400" />
                <h2 className="text-sm font-bold text-slate-200">
                  Supervisor Decision Audit Trail — Cycle {activeTrace?.cycle_id ? activeTrace.cycle_id.slice(0, 8) : 'Current'}
                </h2>
              </div>
              <span className="text-xs font-mono text-slate-400">
                Recorded: {activeTrace ? new Date(activeTrace.recorded_at).toLocaleTimeString() : 'N/A'}
              </span>
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
                <h4 className="text-xs font-mono font-semibold text-slate-400 mb-2 uppercase">1. Session Locks & State</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Supervisor established a <strong className="text-indigo-300">{activeTrace?.details?.session_bias || 'NO_TRADE'}</strong> session bias with {((activeTrace?.details?.bias_confidence || 0) * 100).toFixed(0)}% confidence.
                  Market structure is currently classified under the <strong className="text-indigo-300">{activeTrace?.details?.regime || 'RANGE'}</strong> regime.
                </p>
              </div>

              <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
                <h4 className="text-xs font-mono font-semibold text-slate-400 mb-2 uppercase">2. Worker Consensus</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Technical Agent detected setup: <span className="font-mono text-slate-200">{activeTrace?.details?.technical_analysis?.setup_candidate || 'NO_SETUP'}</span>.
                  ML model gave prediction: <span className="font-mono text-slate-200">{activeTrace?.details?.ml_prediction?.prediction || 'NO_TRADE'}</span>.
                  Validator synthesized all signals and returned: <strong className="text-amber-300">{activeTrace?.details?.validator_decision || 'NO_TRADE'}</strong>.
                </p>
              </div>

              <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4">
                <h4 className="text-xs font-mono font-semibold text-slate-400 mb-2 uppercase">3. Deterministic Risk Guards</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Risk Engine enforces 1% maximum portfolio risk per trade ($100 on $10k equity), 50% max gross exposure, and a 30% unencumbered liquidity reserve buffer. No paper orders are placed without full validator &amp; risk authorization.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: RUNNING POSITIONS & EXECUTED ORDERS */}
      {/* ========================================================================= */}
      {activeTab === 'positions' && (
        <div className="space-y-6">
          {/* Active Open Positions Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <div className="flex items-center justify-between pb-4 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-emerald-400" />
                <h2 className="text-sm font-bold text-slate-200">Active Open Positions ({openPositions.length})</h2>
              </div>
              <span className="text-xs text-slate-400 font-mono">Mark-to-Market Auto-Updated</span>
            </div>

            {openPositions.length === 0 ? (
              <div className="py-12 text-center text-slate-500">
                <Shield className="w-10 h-10 mx-auto mb-3 text-slate-600 opacity-60" />
                <p className="text-sm font-semibold text-slate-400">No Open Paper Positions Active</p>
                <p className="text-xs text-slate-500 max-w-md mx-auto mt-1">
                  The Multi-Agent Supervisor is actively evaluating continuous market cycles. Positions will open automatically when high-probability setups are approved by the Trade Validator and authorized by the Risk Engine.
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto mt-4">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950/60 text-slate-400 font-mono uppercase text-[11px]">
                    <tr>
                      <th className="py-3 px-3">Asset</th>
                      <th className="py-3 px-3">Direction</th>
                      <th className="py-3 px-3">Quantity</th>
                      <th className="py-3 px-3">Entry Price</th>
                      <th className="py-3 px-3">Current Price</th>
                      <th className="py-3 px-3">Stop Loss</th>
                      <th className="py-3 px-3">Take Profit</th>
                      <th className="py-3 px-3">Unrealized P&L</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {openPositions.map((pos) => (
                      <tr key={pos.id} className="hover:bg-slate-800/40">
                        <td className="py-3 px-3 font-bold text-slate-200">{pos.asset}USD</td>
                        <td className="py-3 px-3">
                          <span className={`px-2 py-0.5 rounded font-bold text-[10px] ${
                            pos.direction === 'LONG'
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                              : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                          }`}>
                            {pos.direction}
                          </span>
                        </td>
                        <td className="py-3 px-3 text-slate-300">{pos.quantity}</td>
                        <td className="py-3 px-3 text-slate-300">${Number(pos.entry_price).toFixed(2)}</td>
                        <td className="py-3 px-3 text-slate-200 font-bold">${Number(pos.current_price || pos.entry_price).toFixed(2)}</td>
                        <td className="py-3 px-3 text-rose-400">${Number(pos.stop_loss).toFixed(2)}</td>
                        <td className="py-3 px-3 text-emerald-400">${Number(pos.take_profit).toFixed(2)}</td>
                        <td className={`py-3 px-3 font-bold ${
                          Number(pos.unrealized_pnl) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}>
                          {Number(pos.unrealized_pnl) >= 0 ? '+' : ''}${Number(pos.unrealized_pnl).toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Historical Orders & Executions Table */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <div className="flex items-center justify-between pb-4 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Layers className="w-5 h-5 text-indigo-400" />
                <h2 className="text-sm font-bold text-slate-200">Recent Paper Orders ({orders.length})</h2>
              </div>
              <span className="text-xs text-slate-400 font-mono">Simulated Fills &amp; Slippage</span>
            </div>

            {orders.length === 0 ? (
              <div className="py-8 text-center text-slate-500 text-xs">
                No paper orders placed in the current session yet.
              </div>
            ) : (
              <div className="overflow-x-auto mt-4">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950/60 text-slate-400 font-mono uppercase text-[11px]">
                    <tr>
                      <th className="py-2.5 px-3">Time</th>
                      <th className="py-2.5 px-3">Asset</th>
                      <th className="py-2.5 px-3">Side</th>
                      <th className="py-2.5 px-3">Quantity</th>
                      <th className="py-2.5 px-3">Executed Price</th>
                      <th className="py-2.5 px-3">Fee Paid</th>
                      <th className="py-2.5 px-3">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {orders.map((ord) => (
                      <tr key={ord.id} className="hover:bg-slate-800/40">
                        <td className="py-2.5 px-3 text-slate-400">{new Date(ord.created_at).toLocaleTimeString()}</td>
                        <td className="py-2.5 px-3 font-semibold text-slate-200">{ord.asset}</td>
                        <td className="py-2.5 px-3">
                          <span className={`px-2 py-0.5 rounded font-bold text-[10px] ${
                            ord.side === 'BUY' ? 'text-emerald-400 bg-emerald-500/10' : 'text-rose-400 bg-rose-500/10'
                          }`}>
                            {ord.side}
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-slate-300">{ord.quantity}</td>
                        <td className="py-2.5 px-3 text-slate-200">${Number(ord.executed_price).toFixed(2)}</td>
                        <td className="py-2.5 px-3 text-slate-400">${Number(ord.fee_paid).toFixed(4)}</td>
                        <td className="py-2.5 px-3">
                          <span className="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                            {ord.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: AUDIT DECISION TRACES (Historical Log) */}
      {/* ========================================================================= */}
      {activeTab === 'traces' && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 space-y-4">
          <div className="flex items-center justify-between pb-4 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Database className="w-5 h-5 text-indigo-400" />
              <h2 className="text-sm font-bold text-slate-200">Supervisor Decision Traces Log</h2>
            </div>
            <span className="text-xs text-slate-400 font-mono">PostgreSQL Immutable Audit Trail</span>
          </div>

          <div className="space-y-3">
            {traces.map((trace) => (
              <div
                key={trace.id}
                onClick={() => {
                  setSelectedTraceId(trace.id);
                  setActiveTab('thoughts');
                }}
                className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-indigo-500/50 cursor-pointer transition-all flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-slate-800 text-indigo-400">
                    <Activity className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-slate-200 uppercase font-mono">{trace.details?.asset || 'ASSET'}</span>
                      <span className="text-slate-500">•</span>
                      <span className="text-slate-400 font-mono">{trace.event_type}</span>
                      <span className="px-1.5 py-0.2 rounded text-[10px] bg-slate-800 text-slate-300 font-mono">
                        {trace.to_status}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 mt-1 line-clamp-1">
                      {trace.details?.rationale || `Regime: ${trace.details?.regime || 'range'} | Bias: ${trace.details?.bias_direction || trace.details?.session_bias || 'NO_TRADE'}`}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3 font-mono text-slate-400 text-[11px]">
                  <span>{new Date(trace.recorded_at).toLocaleTimeString()}</span>
                  <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
