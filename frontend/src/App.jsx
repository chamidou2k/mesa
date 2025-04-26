import { useRef, useLayoutEffect, useEffect, useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { Tabs } from 'antd';
import axios from "axios";
import ConfigPanel from './ConfigPanel.jsx';
import InfoPanel from './InfoPanel.jsx';
import GraphPanel from './GraphPanel.jsx';
const BACKEND_API_BASE = import.meta.env.VITE_BACKEND_API_BASE;
function App() {
    const graphContainerRef = useRef(null); // To hold the graph container
    const [graphData, setGraphData] = useState(null);
    const [currModel, setCurrModel] = useState('');
    const [currHwSpec, setCurrHwSpec] = useState('');    
    const [models, setModels] = useState([]);
    const [hwSpecs, setHwSpecs] = useState([]);
    const [CurrNodeId, setCurrNodeId] = useState(0);
    const [activeTab, setActiveTab] = useState(0);

    const runModelEval = async(config) => {
	try {
	    const response = await axios.get(`${BACKEND_API_BASE}/run_model_eval?option=${JSON.stringify(config)}`);
	    setGraphData(response.data.message);
	    console.log("Response:", response.data.message);
	} catch (error) {
	    console.error("Error buildModelApi:", error)
	}
    }

    useLayoutEffect(() => {
	console.log('[DEBUG] MESA Tool Backend API:', BACKEND_API_BASE);
	const fetchAvailProfiles = async() => {
	    try {
		console.log("fetchAvailProfiles:", `${BACKEND_API_BASE}/avail-profiles`);
		const response = await axios.get(`${BACKEND_API_BASE}/avail-profiles`)
		console.log("fetchAvailProfiles: ", response.data.message);
		setModels(response.data.message.models);
		setHwSpecs(response.data.message.hw_specs);
	    } catch (error) {
		console.error("Error fetchAvailProfiles(${BACKEND_API_BASE}/avail-profiles):", error);
	    }
	};
	fetchAvailProfiles();
    }, []);
    const handleConfigChange = (config) => {
	console.log("config change:",  config);
	if (
	    config?.model && 
		config?.hw_spec && 
		models.includes(config.model) && 
		hwSpecs.includes(config.hw_spec)
	) {
	    runModelEval(config);
	} else {
	    console.warn('Invalid config:', {
		validModels: models,
		validHwSpecs: hwSpecs,
		receivedConfig: config
	    });
	}	
    };
    const tabItems = [
	{
	    title: 'Graph',
	    component:(
		<GraphPanel
		    graphData = {graphData}
		    containerId = "graph"
		    CurrNodeId = {CurrNodeId}
		    setCurrNodeId = {setCurrNodeId}
		/>
	    )
	},
	{
	    title: 'Parallel',
	    component:(
		<GraphPanel
		    graphData = {graphData}
		    containerId = "graph"
		    CurrNodeId = {CurrNodeId}
		    setCurrNodeId = {setCurrNodeId}
		/>
	    )
	    
	}
    ];
    return (
	<div className="top_container">
	    <div className="side_pane">
		<ConfigPanel
		    currModel={currModel}
		    setCurrModel={setCurrModel}
		    currHwSpec={currHwSpec}
		    setCurrHwSpec={setCurrHwSpec}
		    models={models}
		    hwSpecs={hwSpecs}
		    onConfigChange={handleConfigChange}
		/>
	    </div>

	    <div className="graph_pane" style={{ height: '100%' }}>
		<div className="tab-headers">
		    {tabItems.map((tab, index) => (
			<button
			    key={index}
			    onClick={() => setActiveTab(index)}
			    className={`tab-header ${activeTab === index ? 'active' : ''}`}
			>
			    {tab.title}
			</button>
		    ))}
		</div>
		<div className="tab-content">
		    {tabItems[activeTab].component}
		</div>
	    </div>
	    <div className="side_pane">
		<InfoPanel
		    graphData = {graphData}
		    CurrNodeId = {CurrNodeId}
		/>

	    </div>
	    
	</div>
    );
}

export default App

