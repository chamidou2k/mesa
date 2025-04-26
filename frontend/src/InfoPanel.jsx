import { useEffect } from 'react';
import "./InfoPanel.css";

const InfoPanel = ({ graphData, CurrNodeId }) => {
    const formatTime = (us) => {
	if (us >= 1e6) {
            return `${(us / 1e6).toFixed(4)} s`;
	} else if (us >= 1e3) {
            return `${(us / 1e3).toFixed(2)} ms`;
	} else if (us >= 1) {
            return `${us.toFixed(2)} us`;
	} else {
            return `${(us * 1e3).toFixed(2)} ns`; 
	}
    };    
    const formatShortType = (bytes) => {
        const units = ['', 'K', 'M', 'G', 'T'];
        for (let i = units.length - 1; i >= 0; i--) {
            if (bytes >= Math.pow(1000, i)) {
                return (bytes / Math.pow(1000, i)).toFixed(4) + ' ' + units[i];
            }
        }
        return bytes.toFixed(4);
    };
    const renderSection = (compData) => {
        if (!compData || !compData.prefill || !compData.regression) return null;

        const allKeys = [
            ...new Set([
                ...Object.keys(compData.prefill),
                ...Object.keys(compData.regression)
            ])
        ];

        return (
            <div className="section-column">

                <div className="metric-table">

                    <div className="table-header">
                        <span className="header-key">Metric</span>
                        <span className="header-value">Prefill</span>
                        <span className="header-value">Regression</span>
                    </div>
                    

                    {allKeys.map((key) => {
			//console.log(`Key: ${key}, Prefill Value:`, compData.prefill[key]);
			return(
                            <div key={key} className="metric-row">
				<span className="metric-key">{key}</span>
				<span className="metric-value">
                                    {compData.prefill[key] !== undefined
				     ? (Array.isArray(compData.prefill[key]) 
					? JSON.stringify(compData.prefill[key])
					: (typeof compData.prefill[key] === 'number' 
					   ? formatShortType(compData.prefill[key])
					   : compData.prefill[key] 
					  )
				       )
                                     : 'N/A'}
				</span>
				<span className="metric-value">
                                    {compData.regression[key] !== undefined
				     ? (Array.isArray(compData.regression[key]) 
					? JSON.stringify(compData.regression[key])
					: (typeof compData.regression[key] === 'number' 
					   ? formatShortType(compData.regression[key])
					   : compData.regression[key] 
					  )
				       )
                                     : 'N/A'}
				</span>
                            </div>
			)})}
                </div>
            </div>
        );
    };
    const renderNestedSection = (compData) => {
        if (!compData || !compData.prefill || !compData.regression) return null;


	const renderNestedData = (data, parentKey = '', depth = 0) => {
            return Object.entries(data).map(([key, value]) => {
		const fullKey = parentKey ? `${parentKey}.${key}` : key;
		const isUtilSubKey = /(^|\.)util\./.test(fullKey); 
		const isLatencyKey = fullKey.toLowerCase().includes('latency');
//		console.log(`renderNestedDate = ${fullKey}, ${isLatencyKey}`);
		const keyStyle = { 
                    paddingLeft: `${depth * 10}px`,
                    minWidth: '70px' 
		};


		const valueStyle = { 
                    paddingLeft: 0,
                    minWidth: '120px' 
		};

		if (typeof value === 'object' && !Array.isArray(value)) {
                    return (
			<div key={fullKey}>

                            <div className="metric-row">
				<span className="metric-key" style={keyStyle}>
                                    {key}
				</span>
				<span className="metric-value" style={valueStyle}>

				</span>
				<span className="metric-value" style={valueStyle}>

				</span>
                            </div>
                            

                            {renderNestedData(value, fullKey, depth + 1)}
			</div>
                    );
		}

		return (
                    <div key={fullKey} className="metric-row">
			<span className="metric-key" style={keyStyle}>{key}</span>
			<span className="metric-value" style={valueStyle}>
                            {renderValue(compData.prefill, fullKey, isLatencyKey, isUtilSubKey)}
			</span>
			<span className="metric-value" style={valueStyle}>
                            {renderValue(compData.regression, fullKey, isLatencyKey, isUtilSubKey)}
			</span>
                    </div>
		);
            });
	};	

	const formatValue = (value, isInsideArray = false, isLatencyKey = false, isUtilSubKey = false) => {
//	    console.log(`formatValue - isLatencyKey = ${isLatencyKey}`);
	    const isLatency = isLatencyKey;// || key.includes('latency');
            if (Array.isArray(value)) {
		return (
                    <span className="array-value">
			[
			{value.map((item, index) => (
                            <span key={index}>
				{formatValue(item, true)} {}
				{index < value.length - 1 && ', '}
                            </span>
			))}
			]
                    </span>
		);
            }

            if (typeof value === 'object' && value !== null) {
		return <span className="object-type">{'{ ... }'}</span>;
            }

            if (typeof value === 'number') {
		//console.log(`isInsideArray-${isInsideArray}`);
		return  isUtilSubKey ?  ((value * 100) > 100 ? <span style={{color: 'red'}}>{(value * 100).toFixed(2)}%</span> : `${(value * 100).toFixed(2)}%`) 
		    : ( isLatency ? formatTime(value) : (isInsideArray ? value : formatShortType(value)));
            }

            return value;
	};


	const renderValue = (dataset, fullKey, isLatencyKey = false, isUtilSubKey = false) => {
            const keys = fullKey.split('.');
            let value;
            
            try {
		value = keys.reduce((acc, k) => acc[k], dataset);
            } catch {
		return 'N/A';
            }

            if (value === undefined) return 'N/A';
            return formatValue(value, false, isLatencyKey, isUtilSubKey);
	};	
  	return (
            <div className="section-column">
		<div className="metric-table">
                    <div className="table-header">
			<span className="header-key">Metric</span>
			<span className="header-value">Prefill</span>
			<span className="header-value">Regression</span>
                    </div>
                    
                    {renderNestedData(compData.prefill)}
		</div>
            </div>
	);
    };
    
    

    const currentNode = graphData?.nodes?.find(item => item.id === CurrNodeId);

    return (
        <div className="info-panel-container">
            <div>
                <h5>Overall Statistics</h5>
                <div className="columns-container">
                    {graphData?.total_comp && (
                        <>
			    {renderNestedSection(graphData?.total_comp)}
			    
                        </>
                    )}
                </div>
            </div>

            {currentNode && (
                <div>
                    <h5>{CurrNodeId}:
			{currentNode?.attributes?.formula && (
			    <span style={{ fontWeight: 'normal', fontSize: '0.7em' }}>
				{currentNode.attributes.formula}
			    </span>
			)}
		    </h5>
                    <div className="columns-container">
                        {currentNode.comp && (
                            <>
                                {renderNestedSection(currentNode.comp)}
                            </>
                        )}
                    </div>
                    <h5>{CurrNodeId} on Device:</h5>
                    <div className="columns-container">
                        {currentNode.dev && (
                            <>
                                {renderNestedSection(currentNode.dev)}
                            </>
                        )}
                    </div>
		    
                </div>
            )}
        </div>
    );
};

export default InfoPanel;
