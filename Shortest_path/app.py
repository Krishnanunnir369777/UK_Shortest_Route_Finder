import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import geopy.distance
import networkx as nx

# -----------------------------
# Load Data
# -----------------------------
data = {'city': ['London','Birmingham','Manchester','Liverpool','Bristol','Newcastle upon Tyne','Sheffield','Cardiff','Leeds','Nottingham','Leicester','Coventry','Bradford','Newcastle','Stoke-on-Trent','Wolverhampton','Derby','Swansea','Plymouth','Reading','Hull','Preston','Luton','Portsmouth','Southampton','Sunderland','Warrington','Bournemouth','Swindon','Oxford','Huddersfield','Slough','Blackpool','Middlesbrough','Ipswich','Telford','York','West Bromwich','Peterborough','Stockport','Brighton','Hastings','Exeter','Chelmsford','Chester','St Helens','Colchester','Crawley','Stevenage','Birkenhead','Bolton','Stockton-on-Tees','Watford','Gloucester','Rotherham','Newport','Cambridge','St Albans','Bury','Southend-on-Sea','Woking','Maidstone','Lincoln','Gillingham','Chesterfield','Oldham','Charlton','Aylesbury','Keighley','Bangor','Scunthorpe','Guildford','Grimsby','Ellesmere Port','Blackburn','Hove','Hartlepool','Taunton','Maidenhead','Aldershot','Great Yarmouth','Rossendale'],
        'latitude': [51.509865,52.4862,53.483959,53.4084,51.4545,54.9784,53.3811,51.4816,53.8008,52.9548,52.6369,52.4068,53.7957,55.007,53.0027,52.5862,52.9228,51.6214,50.3755,51.4543,53.7443,53.7632,51.8787,50.8195,50.9097,54.9069,53.3872,50.7208,51.5686,51.752,53.649,51.5095,53.8175,54.5742,52.0567,52.6784,53.959,52.5187,52.5695,53.4084,50.8225,50.8552,50.7184,51.7361,53.1934,53.4539,51.8892,51.1124,51.9038,53.3934,53.5769,54.5741,51.6562,51.8642,53.432,51.5881,52.2053,51.752,53.591,51.5406,51.3169,51.2704,53.2307,51.3898,53.235,53.5444,51.4941,51.8156,53.867,53.2274,53.5896,51.2362,53.5675,53.2826,53.7486,50.8279,54.6892,51.0143,51.522,51.2484,52.6083,53.6458],
        'longitude': [-0.118092,-1.8904,-2.244644,-2.9916,-2.5879,-1.6174,-1.4701,-3.1791,-1.5491,-1.1581,-1.1398,-1.5197,-1.7593,-1.6174,-2.1794,-2.1288,-1.4777,-3.9436,-4.1427,-0.9781,-0.3326,-2.7031,-0.42,-1.0874,-1.4044,-1.3834,-2.5925,-1.9046,-1.7722,-1.2577,-1.7849,-0.5954,-3.0357,-1.2356,-1.1482,-2.4453,-1.0815,-1.9945,-0.2405,-2.1493,-0.1372,-0.5723,-3.5339,-0.4791,-2.8931,-2.7375,-0.9042,-0.1831,-0.1966,-3.0148,-2.428,-1.3187,-0.39,-2.2382,-1.3502,-3.1409,-0.1218,-0.339,-2.298,0.711,-0.56,-0.5227,-0.5406,-0.5486,-1.4216,-2.1183,-0.068,-0.8084,-1.9064,-4.1297,-0.6544,-0.5704,-0.0802,-2.8976,-2.4877,-0.1688,-1.2122,-3.1036,-0.7205,-0.755,-1.7303,-2.2864]}

df_cities = pd.DataFrame(data).set_index('city')
df_cities["coords"] = list(zip(df_cities["latitude"], df_cities['longitude']))

# -----------------------------------------------------
# Function: Create Graph
# -----------------------------------------------------
def create_graph(df_cities, max_distance_between_nodes):
    graph = {}
    for idx, row in df_cities.iterrows():
        centre_node = idx
        centre_coords = row["coords"]
        df_temp = df_cities[df_cities.index != centre_node].copy()
        df_temp["distance"] = df_temp["coords"].apply(lambda x: geopy.distance.geodesic(x, centre_coords).km)
        df_temp = df_temp[df_temp["distance"] < max_distance_between_nodes]
        graph[centre_node] = dict(zip(df_temp.index, df_temp["distance"]))
    graph = {k: v for k, v in graph.items() if v}
    nodes = list(graph.keys())
    return graph, nodes

# -----------------------------------------------------
# Function: Dijkstra Algorithm
# -----------------------------------------------------
def dijkstra_algorithm(start_node, end_node, nodes, graph):
    unmarked_nodes = nodes.copy()
    shortest_path = {node: np.inf for node in unmarked_nodes}
    shortest_path[start_node] = 0
    previous_nodes = {}
    
    while unmarked_nodes:
        current_marked_node = min(unmarked_nodes, key=lambda node: shortest_path.get(node, float('inf')))
        neighbor_nodes = graph[current_marked_node].keys()
        for neighbor in neighbor_nodes:
            value_on_hold = shortest_path[current_marked_node] + graph[current_marked_node][neighbor]
            if value_on_hold < shortest_path.get(neighbor, float('inf')):
                shortest_path[neighbor] = value_on_hold
                previous_nodes[neighbor] = current_marked_node
        unmarked_nodes.remove(current_marked_node)
    
    path = []
    node = end_node
    while node != start_node:
        path.append(node)
        node = previous_nodes[node]
    path.append(start_node)
    path = list(reversed(path))
    return path, shortest_path[end_node]

# -----------------------------------------------------
# Streamlit App UI
# -----------------------------------------------------
st.set_page_config(page_title="UK Shortest Route Finder", layout="centered")

st.title("🇬🇧 UK Shortest Route Finder (Dijkstra’s Algorithm)")
st.markdown("Find the shortest travel path between two UK cities using Dijkstra’s Algorithm.")

max_distance = st.slider("Max distance between connected cities (km)", 50, 300, 120)
graph, nodes = create_graph(df_cities, max_distance)

start_node = st.selectbox("Select Starting City", options=nodes, index=0)
end_node = st.selectbox("Select Destination City", options=nodes, index=1)

if st.button("Find Shortest Path"):
    try:
        path, distance = dijkstra_algorithm(start_node, end_node, nodes, graph)
        st.success(f"**Shortest path:** {' → '.join(path)}")
        st.info(f"**Total Distance:** {distance:.2f} km")

        # Plot map
        df_plot = pd.DataFrame()
        for row in range(len(path)):
            df_plot.loc[row, "latitude"] = df_cities.loc[path[row], "latitude"]
            df_plot.loc[row, "longitude"] = df_cities.loc[path[row], "longitude"]

        fig = px.scatter_mapbox(df_plot, lat="latitude", lon="longitude", zoom=5, height=500,
                                color_discrete_sequence=["#00355f"], text=path)
        fig.add_trace(px.line_mapbox(df_plot, lat="latitude", lon="longitude").data[0])
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)

    except:
        st.error(f"{start_node} and {end_node} are not connected at this max distance. Try increasing it.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Plotly, and Dijkstra’s Algorithm")
