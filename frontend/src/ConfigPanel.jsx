import {useEffect, useState, useRef, useMemo} from 'react';
import "./ConfigPanel.css";
const ConfigPanel = ({
    currModel,
    setCurrModel,
    currHwSpec,
    setCurrHwSpec,
    models,
    hwSpecs,
    onConfigChange
}) => {
    const defaultParallelConfig = {
	attention: {
	    enabled: true,
	    parallel_strategy: {
		data_parallel: { enabled: true, world_size: 2, level: "node" },
		tensor_parallel: { enabled: true, world_size: 8, level: "gpu" }
	    }
	},
	MLP: {
	    enabled: true,
	    sub_modules: {
		DenseMLP: {
		    enabled: true,
		    parallel_strategy: {
			data_parallel: { enabled: true, world_size: 2, level: "node" },
			tensor_parallel: { enabled: true, world_size: 8, level: "gpu" }
		    }
		},
		SparseMLP: {
		    enabled: true,
		    parallel_strategy: {
			expert_parallel: { enabled: true, world_size: 16, level: "gpu" }
		    }
		}
	    }
	},
	parallel_strategy: {
	    data_parallel: { enabled: false, world_size: 1, level: "node" },
	    tensor_parallel: { enabled: false, world_size: 1, level: "gpu" },
	    pipeline_parallel: { enabled: false, world_size: 1, level: "node" },
	    sequence_parallel: { enabled: false, world_size: 1, level: "gpu" },
	    expert_parallel: { enabled: true, world_size: 16, level: "gpu" }
	}
	
    };


    const [parallelConfig, setParallelConfig] = useState(
	defaultParallelConfig
    );
    const [SeqLength, setSeqLength] = useState(1024);
    const [ContextLength, setContextLength] = useState(1024);
    const [BatchSize, setBatchSize] = useState(1);    
    const [WeightWidth, setWeightWidth] = useState(16);
    const [ActivationWidth, setActivationWidth] = useState(16);
    const [KVWidth, setKVWidth] = useState(16);
    const [StageMode, setStageMode] = useState('Inference');    

    useEffect(() => {
	onConfigChange?.({
	    model: currModel,
	    hw_spec: currHwSpec,
	    stage: StageMode,
	    seq_length: Number(SeqLength),
	    context_length: Number(ContextLength),
	    batch_size: Number(BatchSize),
	    parallel_group: parallelConfig,
	    weight_width: Number(WeightWidth),
	    activation_width: Number(ActivationWidth),
	    kv_width: Number(KVWidth),
	});
    }, [currModel, currHwSpec, StageMode, SeqLength,ContextLength, BatchSize, parallelConfig, WeightWidth, ActivationWidth, KVWidth]);

    useEffect(() => {
        if (hwSpecs.length > 0) {
            setCurrHwSpec(hwSpecs[0]);  
        }
    }, [hwSpecs]);
    useEffect(() => {
        if (models.length > 0) {
            setCurrModel(models[0]);  
        }
    }, [models]);

    const defaultHardwareTopology = {
	accelerator: "H100",
	num_nodes: 1,
	num_gpus: 1,
	inter_connect: "InfiniBand",
	intra_connect: "NVLink"
    };
    const [hardwareTopology, setHardwareTopology] = useState(
	defaultHardwareTopology
    );
    const [parallelType, setParallelType] = useState('non-moe');
    const handleParallelTypeChange = (e) => {
	const value = e.target.value;
	setParallelType(value);
	setParallelConfig(prev => {
	    const newConfig = { ...prev };
	    if (value === 'moe') {
		newConfig.MLP.sub_modules.SparseMLP.enabled = true;
		newConfig.MLP.sub_modules.DenseMLP.enabled = false;
	    } else {
		newConfig.MLP.sub_modules.SparseMLP.enabled = false;
		newConfig.MLP.sub_modules.DenseMLP.enabled = true;
	    }
	    return newConfig;
	});
    };

    const handleToggle = (path) => (checked) => {
	console.log(`${path}`);
	setParallelConfig(prev => {
	    const keys = path.split('.');
	    const newConfig = { ...prev };
	    let current = newConfig;
	    for (let i = 0; i < keys.length - 1; i++) {
		current = current[keys[i]];
	    }
	    current[keys[keys.length - 1]] = checked;
	    return newConfig;
	});
    };


    const handleConfigChange = (path, value) => {
	setParallelConfig(prev => {
	    const keys = path.split('.');
	    const newConfig = { ...prev };
	    let current = newConfig;
	    for (let i = 0; i < keys.length - 1; i++) {
		current = current[keys[i]];
	    }
	    current[keys[keys.length - 1]] = Number(value);
	    return newConfig;
	});
    };


    const totalGPUs = useMemo(() => {
	let multiplier = 1;

	if (parallelConfig.parallel_strategy) {
	    if (parallelConfig.parallel_strategy.data_parallel.enabled) {
		multiplier *= parallelConfig.parallel_strategy.data_parallel.world_size;
	    }
	    if (parallelConfig.parallel_strategy.tensor_parallel.enabled) {
		multiplier *= parallelConfig.parallel_strategy.tensor_parallel.world_size;
	    }
	    if (parallelConfig.parallel_strategy.pipeline_parallel.enabled) {
		multiplier *= parallelConfig.parallel_strategy.pipeline_parallel.world_size;
	    }

	}
	return multiplier;
    }, [parallelConfig]);
    return (
	<div className="config_panel">
	    <h5>Configuration Panel</h5>
	    <div className="config-section compact">
		<label>Models:
		    <select value={currModel} onChange={e => setCurrModel(e.target.value)}>
			<option value="">Select</option>
			{ models.map((model, idx) => <option key={idx} value={model}>{model}</option> )}
		    </select>
		</label>
	    </div>

	    <div className="config-section compact">
		<label>Hardwares:
		    <select value={currHwSpec} onChange={e => setCurrHwSpec(e.target.value)}>
			<option value="">Select</option>
			{ hwSpecs.map((hw, idx) => <option key={idx} value={hw}>{hw}</option> )}
		    </select>
		</label>
	    </div>

	    <div className="config-section compact">
		<fieldset className="stage-select">
		    <legend>Stage</legend>
		    {['Inference', 'Training'].map((stage) => (
			<label key={stage} className="inline-label">
			    <input
				type="radio"
				name="Stage"
				value={stage}
				checked={StageMode === stage}
				onChange={(e) => setStageMode(e.target.value)}
			    />
			    {stage}
			</label>
		    ))}
		</fieldset>
	    </div>

	    <div className="config-section compact">
		<fieldset>
		    <legend>Runtime Parameters</legend>
		    <div className="param-grid">
			<label className="compact-label">
			    Batch Size:
			    <input 
				type="number"
				value={BatchSize}
				onChange={(e) => setBatchSize(Math.max(1, parseInt(e.target.value) || 1))}
				min="1"
				max={4096 * 10}
			    />
			</label>
			<label className="compact-label">
			    Input Length:
			    <input
				type="number"
				value={SeqLength}
				onChange={(e) => setSeqLength(Math.max(1, parseInt(e.target.value) || 1))}
				min="1"
				max={1024 * 1024}
			    />
			</label>
			<label className="compact-label">
			    Output Length:
			    <input
				type="number"
				value={ContextLength}
				onChange={(e) => setContextLength(Math.max(1, parseInt(e.target.value) || 1))}
				min="1"
				max={1024 * 1024}
			    />
			</label>
		    </div>
		</fieldset>
	    </div>

	    <div className="config-section compact">
		<fieldset>
		    <legend>Quantization</legend>
		    <div className="param-grid">
			<label className="compact-label">
			    Weight Width:
			    <select value={WeightWidth} onChange={(e) => setWeightWidth(e.target.value)}>
				<option value="32">32bit</option>
				<option value="16">16bit</option>
				<option value="8">8bit</option>
				<option value="4">4bit</option>
			    </select>
			</label>
			<label className="compact-label">
			    KV Width:
			    <select value={KVWidth} onChange={(e) => setKVWidth(e.target.value)}>
				<option value="32">32bit</option>
				<option value="16">16bit</option>
				<option value="8">8bit</option>
				<option value="4">4bit</option>
			    </select>
			</label>
			<label className="compact-label">
			    Activation Width:
			    <select value={ActivationWidth} onChange={(e) => setActivationWidth(e.target.value)}>
				<option value="32">32bit</option>
				<option value="16">16bit</option>
				<option value="8">8bit</option>
				<option value="4">4bit</option>
			    </select>
			</label>
		    </div>
		</fieldset>
	    </div>
	    <div className="config-section compact">
		<fieldset>
		    <legend>Parallel Group</legend>

		    {/* Module-level toggles for parallel configuration */}
		    
		    <div className="module-toggle">
			<label>Type:

			    <select value={parallelType} onChange={handleParallelTypeChange}>
				<option value="non-moe">Dense MLP</option>
				<option value="moe">Sparse MLP</option>
			    </select>
			</label>
		    </div>
		    

		    <div>
			<div className="strategy-control">
			    <div>
				<label>
				    <input
					type="checkbox"
					readOnly
					checked={parallelConfig.parallel_strategy.data_parallel.enabled}
					onChange={(e) =>
					    handleToggle('parallel_strategy.data_parallel.enabled')(e.target.checked)
					}
				    />
				    Data Parallel:
				    <input
					type="number"
					value={parallelConfig.parallel_strategy.data_parallel.world_size}
					disabled={!parallelConfig.parallel_strategy.data_parallel.enabled}
					onChange={(e) =>
					    handleConfigChange('parallel_strategy.data_parallel.world_size', Math.max(1, e.target.value || 1))
					}
				    />
				</label>
			    </div>
			    <div>

				<label>
				    <input
					type="checkbox"
					checked={parallelConfig.parallel_strategy.tensor_parallel.enabled}
					onChange={(e) =>
					    handleToggle('parallel_strategy.tensor_parallel.enabled')(e.target.checked)
					}
				    />
				    Tensor Parallel:
				    <input
					type="number"
					value={parallelConfig.parallel_strategy.tensor_parallel.world_size}
					disabled={!parallelConfig.parallel_strategy.tensor_parallel.enabled}
					onChange={(e) =>
					    handleConfigChange('parallel_strategy.tensor_parallel.world_size', Math.max(1, e.target.value || 1))
					}
				    />
				</label>
			    </div>
			    <div>

				<label>
				    <input
					type="checkbox"
					readOnly
					checked={parallelConfig.parallel_strategy.pipeline_parallel.enabled}
					onChange={(e) =>
					    handleToggle('parallel_strategy.pipeline_parallel.enabled')(e.target.checked)
					}
				    />
				    Pipeline Parallel:
				    <input
					type="number"
					value={parallelConfig.parallel_strategy.pipeline_parallel.world_size}
					disabled={!parallelConfig.parallel_strategy.pipeline_parallel.enabled}
					onChange={(e) =>
					    handleConfigChange('parallel_strategy.pipeline_parallel.world_size', Math.max(1, e.target.value || 1))
					}
				    />
				</label>
			    </div>
			    <div>

				<label>
				    <input
					type="checkbox"
					readOnly
					checked={parallelConfig.parallel_strategy.sequence_parallel.enabled}
					onChange={(e) =>
					    handleToggle('parallel_strategy.sequence_parallel.enabled')(e.target.checked)
					}
				    />
				    Sequence Parallel:
				    <input
					type="number"
					value={parallelConfig.parallel_strategy.sequence_parallel.world_size}
					disabled={!parallelConfig.parallel_strategy.sequence_parallel.enabled}
					onChange={(e) =>
					    handleConfigChange('parallel_strategy.sequence_parallel.world_size', Math.max(1, e.target.value || 1))
					}
				    />
				</label>

			    </div>
			    {parallelType === "moe" && (
				<div>

				    <label>
					<input
					    type="checkbox"
					    checked={parallelConfig.parallel_strategy.expert_parallel.enabled}
					    onChange={(e) =>
						handleToggle('parallel_strategy.expert_parallel.enabled')(e.target.checked)
					    }
					/>
					Expert Parallel:
					<input
					    type="number"
					    value={parallelConfig.parallel_strategy.expert_parallel.world_size}
					    disabled={!parallelConfig.parallel_strategy.expert_parallel.enabled}
					    onChange={(e) =>
						handleConfigChange('parallel_strategy.expert_parallel.world_size', Math.max(1, e.target.value || 1))
					    }
					/>
				    </label>
				</div>
			    )}
			    
			</div>
			
		    </div>

		    {parallelType === "xmoe" && (
			<div className="module-section">
			    <h4>Attention Module</h4>
			    <div className="strategy-control">
				<div>
				    <label>
					<input
					    type="checkbox"
					    checked={parallelConfig.attention.parallel_strategy.data_parallel.enabled}
					    onChange={(e) =>
						handleToggle('attention.parallel_strategy.data_parallel.enabled')(e.target.checked)
					    }
					/>
					Data Parallel:
					<input
					    type="number"
					    value={parallelConfig.attention.parallel_strategy.data_parallel.world_size}
					    disabled={!parallelConfig.attention.parallel_strategy.data_parallel.enabled}
					    onChange={(e) =>
						handleConfigChange('attention.parallel_strategy.data_parallel.world_size', e.target.value)
					    }
					/>
				    </label>
				</div>
				<div>

				    <label>
					<input
					    type="checkbox"
					    checked={parallelConfig.attention.parallel_strategy.tensor_parallel.enabled}
					    onChange={(e) =>
						handleToggle('attention.parallel_strategy.tensor_parallel.enabled')(e.target.checked)
					    }
					/>
					Tensor Parallel:
					<input
					    type="number"
					    value={parallelConfig.attention.parallel_strategy.tensor_parallel.world_size}
					    disabled={!parallelConfig.attention.parallel_strategy.tensor_parallel.enabled}
					    onChange={(e) =>
						handleConfigChange('attention.parallel_strategy.tensor_parallel.world_size', e.target.value)
					    }
					/>
				    </label>
				</div>
			    </div>
			    <h4>MLP Module</h4>
			    <div className="submodule">
				<div>
				    <label>
					<input
					    type="checkbox"
					    checked={parallelConfig.MLP.sub_modules.DenseMLP.enabled}
					    onChange={(e) =>
						handleToggle('MLP.sub_modules.DenseMLP.enabled')(e.target.checked)
					    }
					/>
					Enable DenseMLP
				    </label>
				</div>
				{parallelConfig.MLP.sub_modules.DenseMLP.enabled && (
				    <div className="strategy-control">
					<div>
					    <label>
						<input
						    type="checkbox"
						    checked={parallelConfig.MLP.sub_modules.DenseMLP.parallel_strategy.data_parallel.enabled}
						    onChange={(e) =>
							handleToggle('MLP.sub_modules.DenseMLP.parallel_strategy.data_parallel.enabled')(e.target.checked)
						    }
						/>
						Data Parallel:
						<input
						    type="number"
						    value={parallelConfig.MLP.sub_modules.DenseMLP.parallel_strategy.data_parallel.world_size}
						    disabled={!parallelConfig.MLP.sub_modules.DenseMLP.parallel_strategy.data_parallel.enabled}
						    onChange={(e) =>
							handleConfigChange('MLP.sub_modules.DenseMLP.parallel_strategy.data_parallel.world_size', e.target.value)
						    }
						/>
					    </label>
					</div>
					<div>
					    <label>
						<input
						    type="checkbox"
						    checked={parallelConfig.MLP.sub_modules.DenseMLP.parallel_strategy.tensor_parallel.enabled}
						    onChange={(e) =>
							handleToggle('MLP.sub_modules.DenseMLP.parallel_strategy.tensor_parallel.enabled')(e.target.checked)
						    }
						/>
						Tensor Parallel:
						<input
						    type="number"
						    value={parallelConfig.MLP.sub_modules.DenseMLP.parallel_strategy.tensor_parallel.world_size}
						    disabled={!parallelConfig.MLP.sub_modules.DenseMLP.parallel_strategy.tensor_parallel.enabled}
						    onChange={(e) =>
							handleConfigChange('MLP.sub_modules.DenseMLP.parallel_strategy.tensor_parallel.world_size', e.target.value)
						    }
						/>
					    </label>
					</div>
				    </div>
				)}
			    </div>
			    <div className="submodule">
				<div>
				    <label>
					<input
					    type="checkbox"
					    checked={parallelConfig.MLP.sub_modules.SparseMLP.enabled}
					    onChange={(e) =>
						handleToggle('MLP.sub_modules.SparseMLP.enabled')(e.target.checked)
					    }
					/>
					Enable MoE
				    </label>
				</div>
				
				{parallelConfig.MLP.sub_modules.SparseMLP.enabled && (
				    <div className="strategy-control">
					<label>
					    <input
						type="checkbox"
						checked={parallelConfig.MLP.sub_modules.SparseMLP.parallel_strategy.expert_parallel.enabled}
						onChange={(e) =>
						    handleToggle('MLP.sub_modules.SparseMLP.parallel_strategy.expert_parallel.enabled')(e.target.checked)
						}
					    />
					    Expert Parallel:
					    <input
						type="number"
						value={parallelConfig.MLP.sub_modules.SparseMLP.parallel_strategy.expert_parallel.world_size}
						disabled={!parallelConfig.MLP.sub_modules.SparseMLP.parallel_strategy.expert_parallel.enabled}
						onChange={(e) =>
						    handleConfigChange('MLP.sub_modules.SparseMLP.parallel_strategy.expert_parallel.world_size', e.target.value)
						}
					    />
					</label>
				    </div>
				)}
			    </div>
			    
			</div>
		    )}


		    <div className="gpu-calculation">
			Total GPUs Required: {totalGPUs}
		    </div>

		</fieldset>
	    </div>
	    
	</div>	
    );
};

export default ConfigPanel;
