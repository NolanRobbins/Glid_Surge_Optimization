"""
Glid Surge Optimization Dashboard
==================================
Interactive dashboard for visualizing port-to-rail optimization.
"""

import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GLID_CLIENTS, US_PORTS, RAIL_TERMINALS, DASHBOARD_CONFIG


import json

# ... imports ...

PAYLOAD_PATH = Path(__file__).parent.parent.parent / "output" / "dashboard_payload.json"

def load_payload():
    """Load the latest dashboard payload."""
    if PAYLOAD_PATH.exists():
        try:
            return json.loads(PAYLOAD_PATH.read_text())
        except Exception:
            return None
    return None

def create_dashboard():
    """Create and configure the Dash application."""
    
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.CYBORG],  # Dark theme
        title="Glid Surge Optimization"
    )
    
    app.layout = create_layout()
    register_callbacks(app)
    
    return app


def create_layout():
    """Create the dashboard layout."""
    
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🚂 Glid First-Mile Surge Optimizer", 
                       className="text-primary mb-0"),
                html.P("Port-to-Rail Optimization System", 
                      className="text-muted")
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Span("Last Updated: ", className="text-muted"),
                    html.Span(id="last-update", className="text-success"),
                ], className="text-end mt-3")
            ], width=4)
        ], className="mb-4"),
        
        # KPI Cards Row
        dbc.Row([
            dbc.Col(create_kpi_card("Containers Today", "container-count", "📦", "primary"), width=3),
            dbc.Col(create_kpi_card("Avg Dwell Time", "dwell-time", "⏱️", "success"), width=3),
            dbc.Col(create_kpi_card("Cost Savings", "cost-savings", "💰", "warning"), width=3),
            dbc.Col(create_kpi_card("Rail Utilization", "rail-util", "🚃", "info"), width=3),
        ], className="mb-4"),
        
        # Main content row
        dbc.Row([
            # Map column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📍 Network Map"),
                    dbc.CardBody([
                        dcc.Graph(id="network-map", style={"height": "500px"})
                    ])
                ])
            ], width=8),
            
            # Alerts column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🚨 Surge Alerts"),
                    dbc.CardBody([
                        html.Div(id="surge-alerts")
                    ])
                ], className="mb-3"),
                dbc.Card([
                    dbc.CardHeader("📅 Optimal Dispatch Windows"),
                    dbc.CardBody([
                        html.Div(id="dispatch-windows")
                    ])
                ])
            ], width=4)
        ], className="mb-4"),
        
        # Charts row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Port Activity Forecast (72h)"),
                    dbc.CardBody([
                        dcc.Graph(id="forecast-chart", style={"height": "300px"})
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("💵 Cost Comparison"),
                    dbc.CardBody([
                        dcc.Graph(id="cost-chart", style={"height": "300px"})
                    ])
                ])
            ], width=6)
        ]),
        
        # Refresh interval
        dcc.Interval(
            id='refresh-interval',
            interval=DASHBOARD_CONFIG['refresh_interval_seconds'] * 1000,
            n_intervals=0
        )
        
    ], fluid=True, className="p-4")


def create_kpi_card(title, value_id, icon, color):
    """Create a KPI card component."""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={"fontSize": "2rem"}),
                html.H2(id=value_id, className=f"text-{color} mb-0 d-inline ms-2"),
            ]),
            html.P(title, className="text-muted mb-0 mt-2")
        ])
    ], className="text-center")


def register_callbacks(app):
    """Register all dashboard callbacks."""
    
    @app.callback(
        [Output("container-count", "children"),
         Output("dwell-time", "children"),
         Output("cost-savings", "children"),
         Output("rail-util", "children"),
         Output("last-update", "children")],
        [Input("refresh-interval", "n_intervals")]
    )
    def update_kpis(n):
        """Update KPI values."""
        # Simulated data - replace with real data
        containers = np.random.randint(150, 250)
        dwell = round(8 + np.random.random() * 8, 1)
        savings = int(15000 + np.random.random() * 10000)
        utilization = int(70 + np.random.random() * 25)
        
        return (
            f"{containers}",
            f"{dwell}h",
            f"${savings:,}",
            f"{utilization}%",
            datetime.now().strftime("%H:%M:%S")
        )
    
    @app.callback(
        Output("network-map", "figure"),
        [Input("refresh-interval", "n_intervals")]
    )
    def update_map(n):
        """Update the network map with active route options."""
        payload = load_payload()
        fig = go.Figure()
        
        # 1. Base Layer: Ports
        port_lats = [p['lat'] for p in US_PORTS]
        port_lons = [p['lon'] for p in US_PORTS]
        port_names = [p['name'] for p in US_PORTS]
        
        fig.add_trace(go.Scattergeo(
            lat=port_lats, lon=port_lons, text=port_names,
            mode='markers+text', marker=dict(size=10, color='#00bcd4', symbol='circle'),
            textposition='top center', textfont=dict(size=9, color='white'),
            name='Ports'
        ))

        # 2. Base Layer: Rail Terminals (sample)
        term_lats = [t['lat'] for t in RAIL_TERMINALS[:20]] # Show top 20 to avoid clutter
        term_lons = [t['lon'] for t in RAIL_TERMINALS[:20]]
        fig.add_trace(go.Scattergeo(
            lat=term_lats, lon=term_lons,
            mode='markers', marker=dict(size=6, color='#ff9800', symbol='square'),
            name='Rail Terminals'
        ))

        # 3. Dynamic Layer: Optimized Routes from GNN
        if payload and "route_options" in payload:
            routes = payload["route_options"].get("routes", [])
            colors = {'road': '#f44336', 'rail': '#4caf50', 'mixed': '#ffeb3b'}
            
            for i, route in enumerate(routes):
                r_type = route.get("metadata", {}).get("routeType", "mixed")
                # Draw segments
                for seg in route.get("segments", []):
                    coords = seg.get("coordinates", [])
                    if not coords: continue
                    lons, lats = zip(*coords)
                    
                    name = f"Route {i+1}: {seg['mode'].upper()}"
                    opacity = 1.0 if i == 0 else 0.5 # Highlight best route
                    width = 4 if i == 0 else 2
                    
                    fig.add_trace(go.Scattergeo(
                        lat=lats, lon=lons, mode='lines',
                        line=dict(width=width, color=colors.get(seg['mode'], 'white')),
                        opacity=opacity, name=name
                    ))

        fig.update_layout(
            geo=dict(
                scope='usa', projection_type='albers usa',
                showland=True, landcolor='rgb(30, 30, 30)',
                countrycolor='rgb(60, 60, 60)', coastlinecolor='rgb(60, 60, 60)',
                showlakes=True, lakecolor='rgb(20, 20, 30)',
                center=dict(lat=38, lon=-95), # Center US
                lataxis_range=[25, 50], lonaxis_range=[-125, -65]
            ),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5, font=dict(color='white'))
        )
        return fig
    
    @app.callback(
        Output("surge-alerts", "children"),
        [Input("refresh-interval", "n_intervals")]
    )
    def update_alerts(n):
        """Update surge alerts based on real GNN payload."""
        payload = load_payload()
        if not payload:
             return html.Div("Waiting for GNN inference...", className="text-muted")

        alerts = []
        # Check 24h horizon predictions
        h24 = payload.get("horizons", {}).get("24", {}).get("port_predictions", {})
        
        for port, surge in h24.items():
            surge_val = float(surge) if surge is not None else 0.0
            if surge_val > 0.7:
                level = "danger"
                msg = f"Critical Surge (Level {surge_val:.2f})"
            elif surge_val > 0.4:
                level = "warning"
                msg = f"Moderate Surge (Level {surge_val:.2f})"
            else:
                level = "success"
                msg = "Optimal Flow"
            
            alerts.append({"port": port, "level": level, "msg": msg})

        # Fallback if empty
        if not alerts:
             alerts.append({"port": "System", "level": "info", "msg": "No port predictions available"})
        
        return html.Div([
            dbc.Alert([
                html.Strong(a['port'] + ": "),
                html.Span(a['msg'])
            ], color=a['level'], className="mb-2 py-2")
            for a in alerts
        ])
    
    @app.callback(
        Output("dispatch-windows", "children"),
        [Input("refresh-interval", "n_intervals")]
    )
    def update_dispatch_windows(n):
        """Update dispatch window recommendations based on Route Options."""
        payload = load_payload()
        if not payload or "route_options" not in payload:
            return html.Div("Calculating optimal windows...", className="text-muted")
            
        routes = payload["route_options"].get("routes", [])
        # Find the best route
        best_route = None
        if routes:
             # Already sorted by score in GNN output
             best_route = routes[0]
        
        if not best_route:
             return html.Div("No routes found", className="text-muted")
             
        score = best_route.get("optimizationScore", 0)
        route_type = best_route.get("metadata", {}).get("routeType", "unknown")
        notes = best_route.get("metadata", {}).get("notes", "Standard dispatch")
        
        # Interpret score as priority
        if score > 80:
             priority = "HIGH PRIORITY"
             color = "success"
        elif score > 50:
             priority = "MEDIUM PRIORITY"
             color = "warning"
        else:
             priority = "DELAY DISPATCH"
             color = "danger"
             
        return html.Div([
            html.Div([
                dbc.Badge(priority, color=color, className="me-2"),
                html.Span(f"Rec: {route_type.upper()} ({score:.0f}/100)", className="fw-bold")
            ], className="mb-2"),
            html.P(notes, className="small text-muted mb-0")
        ])
    
    @app.callback(
        Output("forecast-chart", "figure"),
        [Input("refresh-interval", "n_intervals")]
    )
    def update_forecast(n):
        """Update forecast chart from GNN payload."""
        payload = load_payload()
        fig = go.Figure()
        
        if payload and "horizons" in payload:
            # We have 24, 48, 72h predictions
            # Let's plot the average surge level across all ports for these horizons
            horizons = sorted([int(h) for h in payload["horizons"].keys()])
            avg_surge = []
            
            for h in horizons:
                preds = payload["horizons"][str(h)].get("port_predictions", {}).values()
                valid_preds = [float(p) for p in preds if p is not None]
                avg = np.mean(valid_preds) if valid_preds else 0.5
                avg_surge.append(avg)
            
            # Interpolate for smooth line (0 to 72 hours)
            x_smooth = np.linspace(0, 72, 72)
            if len(horizons) >= 2:
                y_smooth = np.interp(x_smooth, horizons, avg_surge)
            else:
                y_smooth = [avg_surge[0]] * 72 if avg_surge else [0.5] * 72

            fig.add_trace(go.Scatter(
                x=x_smooth, y=y_smooth, mode='lines', name='Avg Network Congestion',
                fill='tozeroy', fillcolor='rgba(0, 188, 212, 0.3)',
                line=dict(color='#00bcd4', width=3)
            ))
        else:
            # Fallback placeholder
            x = list(range(72))
            y = [0.5] * 72
            fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='gray', dash='dash'), name='Waiting for Data'))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Hours Ahead', gridcolor='rgba(255,255,255,0.1)', range=[0, 72]),
            yaxis=dict(title='Congestion Index (0-1)', gridcolor='rgba(255,255,255,0.1)', range=[0, 1]),
            margin=dict(l=40, r=20, t=20, b=40), font=dict(color='white')
        )
        return fig
    
    @app.callback(
        Output("cost-chart", "figure"),
        [Input("refresh-interval", "n_intervals")]
    )
    def update_cost_chart(n):
        """Update cost comparison chart from GNN Route Options."""
        payload = load_payload()
        fig = go.Figure()
        
        road_cost = 0
        intermodal_cost = 0
        
        if payload and "route_options" in payload:
            routes = payload["route_options"].get("routes", [])
            for r in routes:
                r_type = r.get("metadata", {}).get("routeType", "")
                if r_type == "road_only":
                    road_cost = r.get("totalCost", 0)
                elif r_type == "mixed":
                    intermodal_cost = r.get("totalCost", 0)
        
        # Fallbacks if 0 (to show something)
        if road_cost == 0: road_cost = 100
        if intermodal_cost == 0: intermodal_cost = 60
        
        fig.add_trace(go.Bar(
            name='Traditional (Road)', x=['Total Cost'], y=[road_cost],
            marker_color='#f44336'
        ))
        fig.add_trace(go.Bar(
            name='Glid Optimized', x=['Total Cost'], y=[intermodal_cost],
            marker_color='#4caf50'
        ))
        
        fig.update_layout(
            barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='Cost ($)', gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=40, r=20, t=20, b=40), font=dict(color='white'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        return fig


def run_dashboard(debug: bool = True, port: int = None):
    """Run the dashboard application."""
    app = create_dashboard()
    app.run(debug=debug, port=port or DASHBOARD_CONFIG['port'])


if __name__ == "__main__":
    run_dashboard()

