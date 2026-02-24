// Dashboard.jsx — Live Indian Market Indicators
// Fetches: RBI Repo Rate, 10Y G-Sec Yield, NIFTY 50 from backend API
import { useEffect, useState } from "react";
import axios from "axios";

export default function Dashboard() {
  const [market, setMarket] = useState(null);

  useEffect(() => {
    axios.get("http://localhost:8000/api/market-summary")
      .then(r => setMarket(r.data))
      .catch(console.error);
  }, []);

  if (!market) return <div className="loading">Loading live market data...</div>;

  return (
    <div className="dashboard">
      <h1>🇮🇳 Indian Market Dashboard</h1>
      <div className="cards">
        <div className="card">
          <h3>RBI Repo Rate</h3>
          <span className="value">{market.rbi_repo_rate}%</span>
        </div>
        <div className="card">
          <h3>10Y G-Sec Yield</h3>
          <span className="value">{market.gsec_10y_yield}%</span>
        </div>
        <div className="card">
          <h3>NIFTY 50</h3>
          <span className="value">{market.nifty_level?.toLocaleString('en-IN')}</span>
          <span className="sub">{market.nifty_1y_return}% (1Y)</span>
        </div>
        <div className="card">
          <h3>T-Bill Proxy (91D)</h3>
          <span className="value">{market.tbill_proxy_rate}%</span>
        </div>
      </div>
      <p className="updated">Last updated: {market.last_updated}</p>
    </div>
  );
}
