import { useRef,useEffect, useState } from 'react';
import * as G6 from '@antv/g6';
import {register, ExtensionCategory, Graph} from '@antv/g6';
import "./GraphPanel.css";
const GraphPanel = ({graphData, containerId, CurrNodeId, setCurrNodeId}) => {
    const graphRef = useRef(null);
    const containerRef = useRef(null);
    const preSelectedNode = useRef(0);
    const preSelectedNodeColor = useRef(0);
    const [isInitialized, setIsInitialized] = useState(false);

    const clickNode = (node) => {
	console.log(node);
	if (preSelectedNode.current != 0) {
	    graphRef.current.updateNodeData([
		{
		    id: preSelectedNode.current,
		    style: {
			fill: preSelectedNodeColor.current,
		    },
		},
	    ]);
	    
	} 
	preSelectedNode.current = node.id;
	preSelectedNodeColor.current = node.style.fill;
	setCurrNodeId(node.id)

	console.log(node.style.labelText);
	const width = containerRef.current.clientWidth;
	const height = containerRef.current.clientHeight;
	console.log('Click Node Current Container Width:', width);
	console.log('Click Node Current  Container Height:', height);
	graphRef.current.updateNodeData([
	    {
		id: node.id,
		style: {
		    fill: '#00F0F0',
		},
	    },
	]);
	graphRef.current.render();
    };
    useEffect(() => {
	if (!containerId)
	    return;
	const containerIdRef = document.getElementById('graph_panel_id');
	if (!containerIdRef) {
	    cosole.error(`Container with ID "${containerId}" not found`);
	    return;
	}
	console.info(`Container with id"${containerIdRef}"`);
	
	containerRef.current = containerIdRef; 
	console.log("Mounted width:", containerIdRef.clientWidth, ' height:', containerIdRef.clientHeight);

	const MODEL_GRAPH_CONFIG = {
	    container: containerIdRef,
	    width: containerIdRef.clientWidth, 
	    height: containerIdRef.clientHeight,
	    fitView: false,
	    autofit: 'view',
	    node: {
		type: 'rect', 
		style: {
		    size: [100, 20],
		    stroke: '#6fb3e0',
		    badge: true,
		    halo: true,
		    port: true,
		    labelOffsetX: -85,
		    labelOffsetY: -40,
		},
		anchorPoints: [
		    [0.5, 0],
		    [0.5, 1]
		],
		labelCfg: {
		    style: {
			fill: '#2d5c2a',
			fontSize: 8,
			textBaseline: 'middle',
			textAlign: 'center'
		    },
		},	    
	    },
	    edge: {
		type: 'polyline',
		style: {
		    opacity: 0.3,
		    endArrow: true,
		    endArrowType: "triangle",
		    radius: 20, 
		    offset: 50, 
		    stroke: "#000000",
		},
		
	    },

	    layout: {
		type: 'dagre',
		ranksep: 45,
		nodesep: 150,
		align: 'UL',
		rankdir: 'TB',
		controlPoints: true,
		sortByCombo: true,
	    },
	    behaviors: ['drag-canvas', 'zoom-canvas'],
	    groupByTypes: false,
	    combo: {
		type: 'rect',
		position: [0, '50%'], 
		offset: -10,          
		style: {
		    fill: '#00ff00',  
		    stroke: '#ff0000',
		    lineWidth: 0.8, 
		    radius: 6,        
		    opacity: 0.5,   
		},
		labelCfg: {
		    position: 'left',
		    style: {
			fill: '#ff00ff',
			fontSize: 10,   
			fontWeight: 500 
		    }
		},
	    }	    

	};
	const MODEL_GRAPH = new Graph(MODEL_GRAPH_CONFIG);
	graphRef.current = MODEL_GRAPH;
	MODEL_GRAPH.on('node:click', (evt) => {
	    const node = evt.target;
	    clickNode(node);
	});
	MODEL_GRAPH.on('afterrender', () => {
            graphRef.current.fitView();
            setIsInitialized(true);
        });
	const handleResize = () => {
	    if (!graphRef.current || graphRef.current.destroyed) return;
	    const width = containerRef.current.clientWidth;
	    const height = containerRef.current.clientHeight;
	    console.log('handleResize Container Width:', width);
	    console.log('handleResize Container Height:', height);
	    graphRef.current.setSize(width, height);
	    graphRef.current.on('afterlayout', () => {
		console.log("resize afterlayout");
		graphRef.current.fitView();
		const containerHeight = containerRef.current.clientHeight;
		const desiredOffset = containerHeight * 0.5 - containerHeight * 0.1;
	    });
	};
	window.addEventListener('resize', handleResize);
	const resizeObserver = new ResizeObserver((entries) => {
	    for (let entry of entries) {
		const { width, height } = entry.contentRect;
		console.log('New container dimensions:', width, height);
		const widthx = containerRef.current.clientWidth;
		const heightx = containerRef.current.clientHeight;
		console.log('resizeobserver Container Width:', widthx);
		console.log('resizeobserver Container Height:', heightx);
		graphRef.current.setSize(width, height);
		graphRef.current.on('afterlayout', () => {
		    console.log("resizeObserver afterlayout");
		    graphRef.current.fitView();
		    const containerHeight = containerRef.current.clientHeight;
		    const desiredOffset = containerHeight * 0.5 - containerHeight * 0.1;
		});
	    }
	});
	resizeObserver.observe(containerRef.current);	
	return () => {
	    window.removeEventListener('resize', handleResize);
	    resizeObserver.disconnect();	    
	    graphRef.current.destroy();

	}
    }, []);
    useEffect(() => {
	if (!graphData)
	    return;
	const { nodes, edges } = graphData;
	if (preSelectedNode.current != 0) {
	    graphRef.current.updateNodeData([
		{
		    id: preSelectedNode.current,
		    style: {
			fill: preSelectedNodeColor.current,
		    },
		},
	    ]);
	    
	}
	preSelectedNode.current = 0;
	//graphRef.current.clear();
	const width = containerRef.current.clientWidth;
	const height = containerRef.current.clientHeight;
	console.log('[] Container Width:', width);
	console.log('[] Container Height:', height);
	//graphRef.current.setSize(width, height);
	graphRef.current.setData(graphData);
	//console.log("getCombo Data" + JSON.stringify(graphRef.current.getComboData()));
	graphRef.current.render();// Render the graph
	//graphRef.current.fitCenter();
	graphRef.current.on('afterlayout', () => {
	    console.log("graphdata update fit view");
	    graphRef.current.fitView();
	    const containerHeight = containerRef.current.clientHeight;
	    const desiredOffset = containerHeight * 0.5 - containerHeight * 0.1;
	});
    }, [graphData]);
    return (
	<div className="graph_panel"
	     id="graph_panel_id"
	     style={{
		 width: '100%',
		 height: '100%',
	     }}>
	</div>
    );
};

export default GraphPanel;
