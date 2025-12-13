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
        """Update the network map."""
        # Create base figure
        fig = go.Figure()
        
        # Add ports
        port_lats = [p['lat'] for p in US_PORTS]
        port_lons = [p['lon'] for p in US_PORTS]
        port_names = [p['name'] for p in US_PORTS]
        
        fig.add_trace(go.Scattergeo(
            lat=port_lats,
            lon=port_lons,
            text=port_names,
            mode='markers+text',
            marker=dict(size=12, color='#00bcd4', symbol='circle'),
            textposition='top center',
            textfont=dict(size=8, color='white'),
            name='Ports'
        ))
        
        # Add Glid clients
        client_lats = [c['lat'] for c in GLID_CLIENTS.values()]
        client_lons = [c['lon'] for c in GLID_CLIENTS.values()]
        client_names = [c['name'] for c in GLID_CLIENTS.values()]
        
        fig.add_trace(go.Scattergeo(
            lat=client_lats,
            lon=client_lons,
            text=client_names,
            mode='markers+text',
            marker=dict(size=10, color='#4caf50', symbol='diamond'),
            textposition='bottom center',
            textfont=dict(size=8, color='white'),
            name='Glid Clients'
        ))
        
        # Add rail terminals
        terminal_lats = [t['lat'] for t in RAIL_TERMINALS]
        terminal_lons = [t['lon'] for t in RAIL_TERMINALS]
        terminal_names = [t['name'] for t in RAIL_TERMINALS]
        
        fig.add_trace(go.Scattergeo(
            lat=terminal_lats,
            lon=terminal_lons,
            text=terminal_names,
            mode='markers',
            marker=dict(size=8, color='#ff9800', symbol='square'),
            name='Rail Terminals'
        ))
        
        # Update layout
        fig.update_layout(
            geo=dict(
                scope='usa',
                projection_type='albers usa',
                showland=True,
                landcolor='rgb(30, 30, 30)',
                countrycolor='rgb(60, 60, 60)',
                coastlinecolor='rgb(60, 60, 60)',
                showlakes=True,
                lakecolor='rgb(20, 20, 30)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                font=dict(color='white')
            )
        )
        
        return fig
    
    @app.callback(
        Output("surge-alerts", "children"),
        [Input("refresh-interval", "n_intervals")]
    )
    def update_alerts(n):
        """Update surge alerts."""
        alerts = [
            {"port": "Port of LA", "level": "warning", "msg": "Moderate congestion expected 14:00-18:00"},
            {"port": "Port of Long Beach", "level": "success", "msg": "Low congestion - optimal window now"},
            {"port": "Port of Oakland", "level": "danger", "msg": "High volume surge in 4h"},
        ]
        
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
        """Update dispatch window recommendations."""
        now = datetime.now()
        windows = [
            {"time": (now + timedelta(hours=2)).strftime("%H:%M"), "priority": "HIGH", "color": "success"},
            {"time": (now + timedelta(hours=8)).strftime("%H:%M"), "priority": "MED", "color": "warning"},
            {"time": (now + timedelta(hours=14)).strftime("%H:%M"), "priority": "LOW", "color": "secondary"},
        ]
        
        return html.Div([
            dbc.Badge(
                f"{w['priority']}: {w['time']}",
                color=w['color'],
                className="me-2 mb-2 p-2"
            )
            for w in windows
        ])
    
    @app.callback(
        Output("forecast-chart", "figure"),
        [Input("refresh-interval", "n_intervals")]
    )
    def update_forecast(n):
        """Update forecast chart."""
        hours = list(range(72))
        base = 100 + np.sin(np.array(hours) / 12 * np.pi) * 30
        forecast = base + np.random.randn(72) * 10
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=forecast,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(0, 188, 212, 0.3)',
            line=dict(color='#00bcd4')
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Hours Ahead', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='Port Calls', gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=40, r=20, t=20, b=40),
            font=dict(color='white')
        )
        
        return fig
    
    @app.callback(
        Output("cost-chart", "figure"),
        [Input("refresh-interval", "n_intervals")]
    )
    def update_cost_chart(n):
        """Update cost comparison chart."""
        categories = ['Dwell', 'Transport', 'Empty Miles', 'Penalties', 'Energy']
        traditional = [7200, 3500, 2100, 1250, 900]
        glid = [1800, 1800, 600, 250, 225]
        
        fig = go.Figure(data=[
            go.Bar(name='Traditional', x=categories, y=traditional, marker_color='#f44336'),
            go.Bar(name='Glid', x=categories, y=glid, marker_color='#4caf50')
        ])
        
        fig.update_layout(
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(title='Cost ($)', gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=40, r=20, t=20, b=40),
            font=dict(color='white'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        
        return fig


def run_dashboard(debug: bool = True, port: int = None):
    """Run the dashboard application."""
    app = create_dashboard()
    app.run(debug=debug, port=port or DASHBOARD_CONFIG['port'])


if __name__ == "__main__":
    run_dashboard()

