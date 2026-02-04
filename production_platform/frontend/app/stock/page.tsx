"use client";

import { useState, useEffect } from "react";
import api from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { TrendingUp, Activity, Search, LineChart, Cpu } from "lucide-react";
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
    ComposedChart, Line, Bar
} from 'recharts';
import { MotionDiv, fadeIn } from "@/components/ui/motion";

export default function StockPage() {
    const [ticker, setTicker] = useState("AAPL");
    const [history, setHistory] = useState<any[]>([]);
    const [loadingHistory, setLoadingHistory] = useState(false);
    const [predicting, setPredicting] = useState(false);
    const [prediction, setPrediction] = useState<any>(null);
    const [timeframe, setTimeframe] = useState("1 Week");
    const [error, setError] = useState<string | null>(null);

    // Load history
    const fetchHistory = async () => {
        setLoadingHistory(true);
        setError(null);
        try {
            const res = await api.get(`/stock/history?ticker=${ticker}`);
            if (res.data) setHistory(res.data);
        } catch (err: any) {
            setError("Failed to load history. Check ticker.");
        } finally {
            setLoadingHistory(false);
        }
    };

    // Predict
    const handlePredict = async () => {
        setPredicting(true);
        setPrediction(null);
        try {
            const res = await api.post("/stock/predict", {
                ticker,
                timeframe
            });
            setPrediction(res.data);
        } catch (err: any) {
            setError(err.response?.data?.detail || "Prediction failed.");
        } finally {
            setPredicting(false);
        }
    };

    useEffect(() => {
        fetchHistory();
    }, []); // Initial load

    return (
        <div className="space-y-8 max-w-7xl mx-auto">

            {/* Header / Controls */}
            <MotionDiv
                variants={fadeIn}
                initial="hidden"
                animate="visible"
                className="flex flex-col md:flex-row gap-6 items-end justify-between bg-black/20 p-6 rounded-2xl border border-white/5 backdrop-blur-sm"
            >
                <div>
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 to-teal-300">Stock Market Intelligence</h1>
                    <p className="text-muted-foreground mt-1">Real-time LSTM Analysis & Forecasting</p>
                </div>

                <div className="flex flex-col md:flex-row gap-4 w-full md:w-auto items-end">
                    <div className="w-full md:w-48 space-y-2">
                        <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Ticker Symbol</label>
                        <div className="flex gap-2 relative">
                            <Input
                                value={ticker}
                                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                                onKeyDown={(e) => e.key === 'Enter' && fetchHistory()}
                                className="font-mono tracking-wider pl-10 bg-black/40 border-white/10 focus:ring-emerald-500/50"
                            />
                            <Search className="absolute left-3 top-3 h-4 w-4 text-emerald-500" />
                        </div>
                    </div>

                    <div className="w-full md:w-48 space-y-2">
                        <label className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Horizon</label>
                        <select
                            value={timeframe}
                            onChange={(e) => setTimeframe(e.target.value)}
                            className="flex h-11 w-full rounded-lg border border-white/10 bg-black/40 px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/50 transition-all text-white"
                        >
                            <option className="bg-gray-900">1 Day</option>
                            <option className="bg-gray-900">1 Week</option>
                            <option className="bg-gray-900">1 Month</option>
                        </select>
                    </div>

                    <Button
                        onClick={handlePredict}
                        disabled={predicting || loadingHistory}
                        className="w-full md:w-auto h-11 bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/20"
                    >
                        {predicting ? (
                            <>
                                <Cpu className="mr-2 h-4 w-4 animate-pulse" /> Training Neural Net...
                            </>
                        ) : (
                            "Run Prediction Model"
                        )}
                    </Button>
                </div>
            </MotionDiv>

            {error && <MotionDiv className="p-4 bg-red-500/10 text-red-500 rounded-xl border border-red-500/20">{error}</MotionDiv>}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                {/* History Chart */}
                <MotionDiv delay={0.1} className="col-span-1 lg:col-span-2">
                    <Card className="border-white/10 bg-black/40 backdrop-blur-xl">
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2 text-xl">
                                <LineChart className="h-5 w-5 text-emerald-400" />
                                Historical Price & Volume
                                <span className="ml-2 px-2 py-0.5 rounded text-xs bg-white/10 text-muted-foreground font-mono">{ticker}</span>
                            </CardTitle>
                        </CardHeader>
                        <CardContent className="h-[450px]">
                            {loadingHistory ? (
                                <div className="h-full flex items-center justify-center flex-col gap-4 text-muted-foreground">
                                    <div className="h-8 w-8 rounded-full border-2 border-emerald-500 border-t-transparent animate-spin" />
                                    Fetching market data...
                                </div>
                            ) : (
                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart data={history}>
                                        <defs>
                                            <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid stroke="#333" strokeDasharray="3 3" vertical={false} opacity={0.4} />
                                        <XAxis dataKey="date" hide />
                                        <YAxis yAxisId="left" domain={['auto', 'auto']} tick={{ fill: '#6b7280' }} axisLine={false} tickLine={false} />
                                        <YAxis yAxisId="right" orientation="right" hide />
                                        <RechartsTooltip
                                            contentStyle={{ backgroundColor: '#000000cc', borderColor: '#333', borderRadius: '8px', backdropFilter: 'blur(8px)' }}
                                            labelStyle={{ color: '#9ca3af' }}
                                            itemStyle={{ color: '#10b981' }}
                                        />
                                        <Area
                                            yAxisId="left"
                                            type="monotone"
                                            dataKey="close"
                                            stroke="#10b981"
                                            strokeWidth={2}
                                            fill="url(#colorClose)"
                                            activeDot={{ r: 6, fill: "#fff" }}
                                        />
                                        <Bar yAxisId="right" dataKey="volume" fill="#374151" opacity={0.3} barSize={4} />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            )}
                        </CardContent>
                    </Card>
                </MotionDiv>

                {/* Prediction Results */}
                {prediction && (
                    <MotionDiv delay={0.2} className="col-span-1 lg:col-span-2">
                        <Card className="border-emerald-500/30 bg-emerald-500/5 shadow-2xl shadow-emerald-500/10">
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2 text-emerald-400">
                                    <Activity className="h-5 w-5" />
                                    LSTM Model Forecast
                                </CardTitle>
                                <CardDescription>
                                    Bidirectional Long Short-Term Memory Network with Attention Mechanism
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                                    <div className="p-4 bg-black/40 rounded-xl border border-white/5 backdrop-blur text-center">
                                        <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Last Price</div>
                                        <div className="text-3xl font-bold font-mono text-white">${prediction.current_price.toFixed(2)}</div>
                                    </div>
                                    <div className="p-4 bg-black/40 rounded-xl border border-white/5 backdrop-blur text-center">
                                        <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Target</div>
                                        <div className="text-3xl font-bold font-mono text-emerald-400">
                                            ${prediction.predictions[prediction.predictions.length - 1].price.toFixed(2)}
                                        </div>
                                    </div>
                                    <div className="p-4 bg-black/40 rounded-xl border border-white/5 backdrop-blur text-center">
                                        <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Confidence</div>
                                        <div className="text-3xl font-bold font-mono text-emerald-500">{prediction.metrics.confidence.toFixed(1)}%</div>
                                    </div>
                                    <div className="p-4 bg-black/40 rounded-xl border border-white/5 backdrop-blur text-center">
                                        <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Loss</div>
                                        <div className="text-3xl font-bold font-mono text-blue-400">{prediction.metrics.loss.toFixed(5)}</div>
                                    </div>
                                </div>

                                <div className="h-[350px] w-full p-4 bg-black/20 rounded-xl border border-white/5">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AreaChart data={prediction.predictions}>
                                            <defs>
                                                <linearGradient id="predGrade" x1="0" y1="0" x2="0" y2="1">
                                                    <stop offset="5%" stopColor="#34d399" stopOpacity={0.4} />
                                                    <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                                                </linearGradient>
                                            </defs>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#333" opacity={0.5} />
                                            <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                                            <YAxis domain={['auto', 'auto']} tick={{ fill: '#9ca3af', fontSize: 12 }} />
                                            <RechartsTooltip
                                                contentStyle={{ backgroundColor: '#064e3b', borderColor: '#059669', color: '#fff' }}
                                                itemStyle={{ color: '#fff' }}
                                            />
                                            <Area
                                                type="monotone"
                                                dataKey="price"
                                                stroke="#34d399"
                                                strokeWidth={3}
                                                fill="url(#predGrade)"
                                                activeDot={{ r: 8, strokeWidth: 0, fill: '#fff' }}
                                            />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                            </CardContent>
                        </Card>
                    </MotionDiv>
                )}
            </div>
        </div>
    );
}
