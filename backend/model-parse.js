const axios = require('axios');
const fs = require('fs');
const {model_eval} = require("./model-eval.js");
const CONFIG_BASE_PATH = "../config";

function loadAvailModels() {
    try {
	console.log("loadAvailHWSpecs");
	const data = fs.readFileSync(`${CONFIG_BASE_PATH}/models/models-list.json`, 'utf8');
	console.log(data); 
	return JSON.parse(data).models;
    } catch (err) {
	console.error("Error reading or pasring the file:", err);
	return null;
    }
} 
function loadAvailHWSpecs() {
    try {
	console.log("loadAvailHWSpecs");
	const data = fs.readFileSync(`${CONFIG_BASE_PATH}/hardware/hw-list.json`, 'utf8');
	console.log(data); 
	return JSON.parse(data).hardwares;
    } catch (err) {
	console.error("Error loadAvailHWSpecs from(hw-specs-list.json):", err);
	return null;
    }
} 

function load_model_parameters(model_para_file) {
    try {
	const data = fs.readFileSync(model_para_file, 'utf8');
	return JSON.parse(data);
    } catch (err) {
	console.error("Error reading or pasring the file:", err);
	return null;
    }
}
function load_hw_parameters(hw_para_file) {
    try {
	const data = fs.readFileSync(hw_para_file, 'utf8');
	return JSON.parse(data);
    } catch (err) {
	console.error("Error reading or pasring the file:", err);
	return null;
    }
}

function read_model_graph(model_file_path) {
    try {
	const data = fs.readFileSync(model_file_path, 'utf8');
	return JSON.parse(data);
    } catch (err) {
	console.error("Error reading or pasring the file:", err);
	return null;
    }
}
function build_graph_data(json_data) {
    const nodes = [];
    const edges = [];
    const comboSet = new Set();

    json_data.graph.nodes.forEach((node) => {
        if (node.combo) 
            comboSet.add(node.combo);
	//console.log(node.formula);
	nodes.push({
	    id: node.name,  
	    attributes: {
		op_type: node.op_type, 
		formula: node.formula,
		inputs: node.inputs,
		outputs: node.outputs,
	    },
	    combo: node.combo,
	    style: {
		labelText: node.name+'('+node.op_type+')',
		op_type: node.op_type    
	    },
	    
	});
	
	node.inputs.forEach((input) => {
	    edges.push({
		id: input + '->' + node.name,
		source: input,           
		target: node.name,       
		style: {
		}
	    });
	});
    });
    const combos = Array.from(comboSet).map(comboId => ({
        id: comboId,
        style: {
            labelText: comboId
        }
    }));

    return {
	attributes: {
	    name: 'GraphTopo'
	},
	nodes: nodes,
	edges: edges,
	combos: combos
    };
    
}
function load_model_graph(model_name) {
    console.log(model_name);
    model_name = model_name.replace(/\//g, ".");
    model_para = load_model_parameters(`${CONFIG_BASE_PATH}/models/${model_name}.json`);
    console.log(model_para);
    model_arch_name = model_para.architectures[0];
    ret = read_model_graph(`${CONFIG_BASE_PATH}/models/arch/${model_arch_name}.json`);
    console.log(JSON.stringify(ret));
    graph_data = build_graph_data(ret);
    const context_para = {
	seq_length: 1024,
	batch_size: 1,
	tp: 1,
	sp: 1,
	pp: 1,
	dp: 1,
	cp: 1,
	ep: 1,
	stage: "Prefill",
	a_width: 2,
	kv_width: 2,
	w_width: 2
	
    };
    const hw_spec = {
	flops: 989 * Math.pow(10,12),
	bandwidth: 3.352 * Math.pow(10, 12),
	bw_ratio: 0.912,
    };
    console.log(hw_spec);
    model_eval(graph_data, model_para, context_para, hw_spec);
    console.log(graph_data);
    return graph_data;
}
function convertMemoryCapacity(memoryStr) {
    const match = memoryStr.match(/(\d+)([a-zA-Z]+)/);
    if (!match) {
	throw new Error("Invalid memory capacity format");
    }
    const number = parseInt(match[1], 10); 
    const unit = match[2].toUpperCase();   
    let memory;
    if (unit === "GB") {
	memory = number * Math.pow(10, 9);
    } else if (unit === "MB") {
	memory = number * Math.pow(10, 6); 
    } else {
	throw new Error("Unsupported memory unit: " + unit);
    }
    return memory;
}
function convertBandwidth(bwStr) {
  const match = bwStr.match(/(\d+(\.\d+)?)([a-zA-Z]+)/);
  if (!match) {
    throw new Error("Invalid bandwidth format");
  }
  const number = parseFloat(match[1]);
  const unit = match[3].toUpperCase();
  let bw;
  if (unit === "TBPS") {
    bw = number * Math.pow(10, 12);
  } else if (unit === "GBPS") {
    bw = number * Math.pow(10, 9);
  } else {
    throw new Error("Unsupported bw unit: " + unit);
  }
  return bw;
}
function calculateFLOPS(hw_spec_para, precision) {
    const sparsitySupported = {
	"FP64": hw_spec_para.FP64_sparsity,
	"TF64": hw_spec_para.TF64_sparsity,
	"FP32": hw_spec_para.FP32_sparsity,
	"TF32": hw_spec_para.TF32_sparsity,
	"FP16": hw_spec_para.FP16_sparsity,
	"BF16": hw_spec_para.BF16_sparsity,
	"FP8": hw_spec_para.FP8_sparsity,
	"INT8": hw_spec_para.INT8_sparsity,
	"FP4": hw_spec_para.FP4_sparsity
    };

    const flopsMapping = {
	"FP64": hw_spec_para.FP64_TFLOPS,
	"TF64": hw_spec_para.TF64_TFLOPS,
	"FP32": hw_spec_para.FP32_TFLOPS,
	"TF32": hw_spec_para.TF32_TFLOPS,
	"FP16": hw_spec_para.FP16_TFLOPS,
	"BF16": hw_spec_para.BF16_TFLOPS,
	"FP8": hw_spec_para.FP8_TFLOPS,
	"INT8": hw_spec_para.INT8_TOps,
	"FP4": hw_spec_para.FP4_TFLOPS
    };

    let totalFLOPS = 0;

    let t_ops = flopsMapping[precision];
    if (t_ops !== null) {
	if (sparsitySupported[precision]) {
            t_ops /= 2;
	}
    }

    return t_ops * Math.pow(10, 12);
}
function run_model_eval(config) {

    console.log(JSON.stringify(config));
    model_name = config.model;
    let parallel_group = config.parallel_group;
    let hw_spec_name = config.hw_spec;
    console.log(`parallel_group: ${parallel_group}`)
    model_name = model_name.replace(/\//g, ".");
    model_para = load_model_parameters(`${CONFIG_BASE_PATH}/models/${model_name}.json`);
    hw_spec_name = hw_spec_name.replace(/\//g, ".");
    hw_spec_para = load_hw_parameters(`${CONFIG_BASE_PATH}/hardware/${hw_spec_name}.json`);
    console.log(`model: ${model_para}, hw_spec: ${hw_spec_name}`);
    console.log(JSON.stringify(hw_spec_para));

    model_arch_name = model_para.architectures[0];
    ret = read_model_graph(`${CONFIG_BASE_PATH}/models/arch/${model_arch_name}.json`);

    graph_data = build_graph_data(ret);
    const context_para = {
	seq_length: config.seq_length,
	context_length: config.context_length,
	batch_size: config.batch_size,
	tp: parallel_group.parallel_strategy.tensor_parallel.enabled ? parallel_group.parallel_strategy.tensor_parallel.world_size : 1,
	sp: parallel_group.parallel_strategy.sequence_parallel.enabled ? parallel_group.parallel_strategy.sequence_parallel.world_size : 1,
	pp: parallel_group.parallel_strategy.pipeline_parallel.enabled ? parallel_group.parallel_strategy.pipeline_parallel.world_size : 1,
	dp: parallel_group.parallel_strategy.data_parallel.enabled ? parallel_group.parallel_strategy.data_parallel.world_size : 1,
	cp: 1,
	ep: parallel_group.parallel_strategy.expert_parallel.enabled ? parallel_group.parallel_strategy.expert_parallel.world_size : 1,
	
	stage: config.stage,
	a_width: config.activation_width / 8,
	kv_width: config.kv_width / 8,
	w_width: config.weight_width / 8
	
    };

    const hw_spec = {
	flops:  calculateFLOPS(hw_spec_para, "FP16"), //hw_spec_para.FP16_TFLOPS * Math.pow(10,12), //989T
	mem_size: convertMemoryCapacity(hw_spec_para.Tier1_memory_capacity), //80 * Math.pow(10, 9), 
	mem_bw: convertBandwidth(hw_spec_para.Tier1_memory_bandwidth), //3.352 * Math.pow(10, 12), //3.2T
	mem_bw_util: 0.9,
	scaleup_bw: convertBandwidth(hw_spec_para.scaleup_link_bandwidth),//900 * Math.pow(10, 9), //900GB
	scaleup_bw_util: 0.99, //hardcode
	scaleout_bw: 400 * Math.pow(10, 9) / 8, //hardcode
	scaleout_bw_util: 0.98, // hard-corde
	scaleout_delay: 3, //hard-code now, 3us for RDMA, 10us for TCP-IP
    };
    console.log(`Context Parameters: ${JSON.stringify(context_para, null, 2)}`);
    console.log(`HW Spec: ${JSON.stringify(hw_spec, null, 2)}`);    

    model_eval(graph_data, model_para, context_para, hw_spec);
    return graph_data;
}

function _avail_models() {
    return model_list;
}
module.exports = {
    load_model_graph,
    loadAvailHWSpecs,
    loadAvailModels,
    run_model_eval,
    build_graph_data,
    load_model_parameters,
    read_model_graph
};
