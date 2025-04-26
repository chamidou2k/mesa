const { op_eval } = require("./op-calc.js");
const math  = require('mathjs');
function model_eval(model_topo, model_para, context_para, hw_spec) {
    const phases = ['prefill', 'regression'];
    const num_layers = model_para.num_hidden_layers;
    let moe_num_layers = 0;
    if (model_para.first_k_dense_replace !== undefined) {
	moe_num_layers =  num_layers - model_para.first_k_dense_replace;
    }
    const dimension = model_para.hidden_size;
    const ffn_dimension = model_para.intermediate_size || model_para.ffn_dim;
    const h_heads = model_para.head_dim || model_para.num_attention_heads;
    const kv_heads = model_para.num_key_value_heads || h_heads;
    const vocab_size = model_para.vocab_size;
    const total_gpu = context_para.tp * context_para.pp * context_para.dp;
    let n_local_experts = 1, n_local_shared_experts = 1; // local experts number in each devicee
    if (context_para.ep > 1 && model_para.n_routed_experts !== undefined) 
	n_local_experts = model_para.n_routed_experts / context_para.ep;
    if (context_para.ep > 1 && model_para.n_shared_experts !== undefined) 
	n_local_shared_experts = Math.max(1, Math.floor(model_para.n_shared_experts / context_para.ep));
    const total_comp = {
        prefill: { flops: 0, memory: 0, latency: 0 },
        regression: { flops: 0, memory: 0, latency: 0 }
    };
    const total_combo = {
	prefill: {},
	regression: {}
    };
	const gpu_stats = {
        prefill: [],
        regression: []
    };

    phases.forEach(phase => {
        const ctx = context_para;
        let phase_flops = 0;
        let phase_memory = 0;
        let phase_latency = 0;
	let layer_flops = 0;
        let layer_memory = 0;
        let layer_latency = 0;
	let layer_total_memory = 0;
	let layer_mem_kv = 0;
	let layer_mem_weight = 0;
	let nonlayer_mem_weight = 0;
	let nonlayer_flops = 0;
        let nonlayer_memory = 0;
        let nonlayer_latency = 0;
	let nonlayer_total_memory = 0;	
	let comboComps = new Map();
	let comboDevs = new Map();

	let acts_mem_peak = 0;
	let layer_comm_count = context_para.pp;
	let layer_comm_latency = 0;
	model_topo.nodes.forEach(node => {
	    node.comp = node.comp || {};
	    node.dev = node.dev || {};
	    let targetNode = null;
	    let pre_shape = null;
	    if (node.attributes.inputs && node.attributes.inputs.length > 0) {
		const shapes = node.attributes.inputs.map(inputId => {
		    const targetNode = model_topo.nodes.find(n => n.id === inputId);
		    if (!targetNode || !targetNode.dev[phase] || !targetNode.dev[phase].out_shape) {
			console.warn(`Not able to find valid in_shape "${inputId}"`);
			return null;
		    }
		    const outShape = targetNode.dev[phase].out_shape;
		    console.log(`outShape: ${JSON.stringify(outShape)}`);
		    const shapeForInput = outShape.hasOwnProperty(node.id) ? outShape[node.id] : outShape;
		    console.log(`shapeForInput: ${JSON.stringify(shapeForInput)}`);
		    return JSON.parse(JSON.stringify(shapeForInput));
		});
		pre_shape = node.attributes.inputs.length === 1 ? shapes[0] : shapes;
	    } else {
		pre_shape = null;
	    }
	    
	    const {node_comp: comp, dev_comp: dev} = calculateNodeMetrics(node,  ctx, { dimension, ffn_dimension, h_heads, kv_heads, vocab_size, hw_spec, pre_shape }, model_para, phase); 
	    node.comp[phase] = comp;
	    node.dev[phase] = dev;
	    if (comp.mem.acts > acts_mem_peak)
		acts_mem_peak = comp.mem.acts;
	    if (node.combo) {

		if (!comboComps.has(node.combo)) {
		    comboComps.set(node.combo, {
			latency: 0,
			flops: 0,
			mem: {
			    access: 0,
			    total: 0,
			    acts: 0,
			    kv: 0,
			    weight: 0,
			},
			nodes: []
		    });

		}

		let comboComp = comboComps.get(node.combo);
		comboComp.flops += comp.flops;
		comboComp.mem.access += comp.mem.access;
		comboComp.mem.total += comp.mem.peak;
		comboComp.mem.acts += comp.mem.acts;
		comboComp.mem.kv += comp.mem.kv;
		comboComp.mem.weight += comp.mem.weight;
		comboComp.latency += comp.latency.total;
		comboComp.nodes.push(node.id);

		if (!comboDevs.has(node.combo)) {
		    comboDevs.set(node.combo, {
			latency: 0,
			flops: 0,
			mem: {
			    access: 0,
			    total: 0,
			    acts: 0,
			    kv: 0,
			    weight: 0,
			},
			nodes: []
		    });

		}
		let comboDev = comboDevs.get(node.combo);
		comboDev.flops += dev.flops;
		comboDev.mem.access += dev.mem.access;
		comboDev.mem.total += dev.mem.peak;
		comboDev.mem.acts += dev.mem.acts;
		comboDev.mem.kv += dev.mem.kv;
		comboDev.mem.weight += dev.mem.weight;
		comboDev.latency += dev.latency.total;
		comboDev.nodes.push(node.id);
		//excluded routed, shared, and dense MLP 
		if (node.combo !== "Routed" && node.combo !== "Shared" && node.combo !== "Dense-MLP") {
		    layer_flops += comp.flops;
		    layer_memory += comp.mem.access;
		    layer_total_memory += comp.mem.peak;
		    layer_latency += comp.latency.total;
		    layer_mem_kv += comp.mem.kv;
		    layer_mem_weight += comp.mem.weight;
		}
	    } else {
		nonlayer_flops += comp.flops;
		nonlayer_memory += comp.mem.access;
		nonlayer_total_memory += comp.mem.peak;
		nonlayer_latency += comp.latency.total;
		nonlayer_mem_weight += comp.mem.weight;
	    }
	    
	});
	let routed_device = {
	    mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
	    flops: 0,
	    latency: 0
	};
	
	let shared_device = {
	    mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
	    flops: 0,
	    latency: 0
	};
	let input_device = {
	    mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
	    flops: 0,
	    latency: 0
	};
	let output_device = {
	    mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
	    flops: 0,
	    latency: 0
	};
	let densemlp_device = {
		mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
		flops: 0,
		latency: 0
	};
	let ffn_device = {
	    mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
	    flops: 0,
	    latency: 0
	}
	let attn_device = {
		mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
		flops: 0,
		latency: 0
	};
	let moe_device = {
	    mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
	    flops: 0,
	    latency: 0
	}
	    
	const fillComp = (device, comp) => {
		device.flops = comp.flops;
		device.latency = comp.latency;
		Object.keys(device.mem).forEach(key => {
			device.mem[key] = comp.mem[key];
		});
	};
	
	const mergeComp = (a, b) => {
	    a.flops += b.flops;
	    a.latency += b.latency;
	    Object.keys(a.mem).forEach(key => {
		a.mem[key] += b.mem[key];
	    });
	};
	const scaleComp = (a, scale) => {
	    Object.keys(a.mem).forEach(key => {
		a.mem[key] *= scale;       
	    });
	    a.flops   *= scale;   
	    a.latency *= scale;
	};

	if (comboDevs.has("Input")) {
		const input = comboDevs.get("Input");
		fillComp(input_device, input);
	}

	if (comboDevs.has("Output")) {
		const output = comboDevs.get("Output");
		fillComp(output_device, output);
	}

	
	if (comboDevs.has("Dense-MLP")) {
		const dm = comboDevs.get("Dense-MLP");
		fillComp(densemlp_device, dm);
	}
	if (comboDevs.has("FFN")) {
		const ffn = comboDevs.get("FFN");
		fillComp(ffn_device, ffn);
	}
	if (comboDevs.has("ATTN")) {
		const attn = comboDevs.get("ATTN");
		fillComp(attn_device, attn);
	}
	if (comboDevs.has("MoE")) {
		const moe = comboDevs.get("MoE");
		fillComp(moe_device, moe);
	}

	comboDevs.forEach((value, key) => {
	    const scale = key === "Routed" ? n_local_experts :
                  key === "Shared" ? n_local_shared_experts : 1;
	    
	    if (key === "Routed" || key === "Shared") {
		const target = key === "Routed" ? routed_device : shared_device;
		
		for (const memKey in target.mem) {
		    target.mem[memKey] = (value.mem?.[memKey] || 0) * scale;
		}
		
		target.flops = (value.flops || 0) * scale;
		target.latency = (value.latency || 0) * scale;
	    }
	});
	
	const expert_device = {
	    mem: {
		total: routed_device.mem.total + shared_device.mem.total,
		weight: routed_device.mem.weight + shared_device.mem.weight,
		kv: routed_device.mem.kv + shared_device.mem.kv,
		access: routed_device.mem.access + shared_device.mem.access,
		acts: routed_device.mem.acts + shared_device.mem.acts
	    },
	    flops: routed_device.flops + shared_device.flops,
	    latency: routed_device.latency + shared_device.latency
	};
	mergeComp(moe_device, expert_device);
	/*
	console.log(`routed: ${JSON.stringify(routed_device)}`);
	console.log(`shared: ${JSON.stringify(shared_device)}`);
	console.log(`expert: ${JSON.stringify(expert_device)}`);
	console.log(`attn_device: ${JSON.stringify(attn_device)}`);
	console.log(`densemlp_device: ${JSON.stringify(densemlp_device)}`);
	console.log(`ffn_device: ${JSON.stringify(ffn_device)}`);
	console.log(`moe_device: ${JSON.stringify(moe_device)}`);
	console.log(`input_device: ${JSON.stringify(input_device)}`);
	console.log(`output_device: ${JSON.stringify(output_device)}`);
	*/
	let moe_per_dev = {
		mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
		flops: 0,
		latency: 0
	};
	let densemlp_per_dev = {
		mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
		flops: 0,
		latency: 0
	};
	let ffn_per_dev = {
	    mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
	    flops: 0,
	    latency: 0
	};
	let attn_per_dev = {
	    mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
	    flops: 0,
	    latency: 0
	};
	let total_per_dev = {
	    mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
	    flops: 0,
	    latency: 0
	};
	let total_comp_final = {
	    mem: { total: 0, weight: 0, kv: 0, access: 0, acts: 0 },
	    flops: 0,
	    latency: 0
	}
	const moe_num_layerer = Math.floor(moe_num_layers  / context_para.pp);
	fillComp(moe_per_dev, moe_device);
	scaleComp(moe_per_dev, moe_num_layers);
	fillComp(densemlp_per_dev, densemlp_device);
	scaleComp(densemlp_per_dev, num_layers - moe_num_layers);
	fillComp(ffn_per_dev, ffn_device);
	scaleComp(ffn_per_dev, num_layers);
	mergeComp(ffn_per_dev, moe_per_dev);
	mergeComp(ffn_per_dev, densemlp_per_dev);
	fillComp(attn_per_dev, attn_device);
	scaleComp(attn_per_dev, num_layers);
	mergeComp(total_per_dev, attn_per_dev);
	mergeComp(total_per_dev, ffn_per_dev);
	mergeComp(total_per_dev, input_device);
	mergeComp(total_per_dev, output_device);
	fillComp(total_comp_final, total_per_dev);
	scaleComp(total_comp_final, total_gpu);
	total_comp_final.latency /= total_gpu;
	/* commented out for debug
	console.log(`ffn_per_dev: ${JSON.stringify(ffn_per_dev)}`);
	console.log(`attn_per_dev: ${JSON.stringify(attn_per_dev)}`);
	console.log(`moe_per_dev: ${JSON.stringify(moe_per_dev)}`);
	console.log(`densemlp_per_dev: ${JSON.stringify(densemlp_per_dev)}`);
	console.log(`total_per_dev: ${JSON.stringify(total_per_dev)}`);
	console.log(`total_comp_final: ${JSON.stringify(total_comp_final)}`);
	console.log(`${hw_spec.mem_size}, ${(nonlayer_total_memory + layer_total_memory * num_layers + acts_mem_peak)}`);
	*/
        total_comp[phase] = {
	    latency: total_comp_final.latency,
            flops: total_comp_final.flops,
            mem: {
		access: total_comp_final.mem.access,
		total: total_comp_final.mem.total,
		acts: acts_mem_peak,
		kv: total_comp_final.mem.kv,
		weight: total_comp_final.mem.weight,
	    },
	    util: {
		mem_size: total_comp_final.mem.total / (hw_spec.mem_size * total_gpu),
		mem_bw:  total_comp_final.mem.access / (hw_spec.mem_bw * total_gpu),
		flops: total_comp_final.flops / (hw_spec.flops * total_gpu),
		up_bw: 0,
		out_bw: 0,
	    },
	    intensity: (total_comp_final.flops) / (total_comp_final.mem.access),
	    roofline: hw_spec.flops / hw_spec.mem_bw,
	    througput: 1e6/(total_comp_final.latency) * ctx.batch_size,
            };

    });
    
    model_topo.total_comp = total_comp;
    return model_topo;
}


function formula_eval(formula, para) {
    let matchResult = formula.match(/\[(.*?)\]/); 
    if (matchResult) {
	let paramsStr = matchResult[1]; 
	console.log(paramsStr);
	let paramNames = paramsStr.split(/,\s*/); 
	console.log(paramNames);
	return [math.evaluate(paramNames[0], para), math.evaluate(paramNames[1], para)];
    } else {
	return [0, 0];
    }	
}
function outputs_eval(outputs) {
    let outputList = [];
    if (typeof outputs === 'string') {

        if (outputs.trim() === '') {
            console.warn('Input is an empty string');
            return [];
        }
        const matchResult = outputs.match(/^\s*$$(.*)$$$/);
        if (matchResult) {
            outputList = matchResult[1].split(/\s*,\s*/)
                                       .map(item => item.trim())
                                       .filter(item => item !== '');
        } else {
            outputList = [outputs.trim()];
        }
    } else if (Array.isArray(outputs)) {
        outputList = outputs.filter(item => typeof item === 'string' && item.trim() !== '');
    } else {
        const str = String(outputs).trim();
        outputList = str ? [str] : [];
        console.warn(`Non-string/array input converted to: ${JSON.stringify(outputList)}`);
    }
    console.log('Parsed output list:', outputList);
    return outputList;
}
function outputs_eval_old(outputs) {
    console.log(`outputs: ${outputs}`);
    let outputList = [];
    if (typeof outputs === 'string') {
	let matchResult = outputs.match(/\[(.*?)\]/);
	if (matchResult) {
	    outputList = matchResult[1].split(/\s*,\s*/).filter(item => item.trim() !== '');
	} else {
	    outputList = outputs.trim() ? [outputs.trim()] : [];
	}
    } else if (Array.isArray(outputs)) {
	outputList = outputs;
    }
    console.log('Parsed output list:', outputList);
    return outputList;
}
function _init_comp() {
    return {
	flops: 0,
	in_shape: [],
	w_shape: [],
	out_shape: [],
	mem: {
	    access: 0,
	    peak: 0, 
	    acts: 0, 
	    kv: 0, 
	    weight: 0, 
	},	
	latency: {
	    total: 0,
	    mem:  0, 
	    comp: 0,
	    comm: 0,
	},
	comm_pattern: 'NA',
	comm_para: 0,
	bound: 'compute',
	ops2bytes: 0,
	
    };
}

function calculateNodeMetrics(node, ctx, params, model_para, phase) {
    let comp_old = {
        flops: 0,
        load_activation: 0,
        load_weight: 0,
        load_kv: 0,
        store_activation: 0,
        store_kv: 0,
        total_memory: 0,
        memory_latency: 0,
        compute_latency: 0,
	comm_latency: 0,
        latency: 0,
	comm_pattern: 'NA',
	comm_para: 0,
        ops2bytes: 0,
        bound: 'compute',
	in_shape: [],
	out_shape: [],
	w_shape: []
    };
    let dev_old = {
        flops: 0,
        load_activation: 0,
        load_weight: 0,
        load_kv: 0,
        store_activation: 0,
        store_kv: 0,
        total_memory: 0,
        memory_latency: 0,
        compute_latency: 0,
        latency: 0,
        ops2bytes: 0,
        bound: 'compute',
	in_shape: [],
	out_shape: [],
	w_shape: []
    };
    let dev_comp = _init_comp();
    let node_comp = _init_comp();
    
    const { dimension, ffn_dimension, h_heads, kv_heads, hw_spec, pre_shape } = params;
    const [batch, seq_len, context_len] = [ctx.batch_size, ctx.seq_length, ctx.context_length];
    const [vocab_size] = [params.vocab_size];
    const op_type = node.attributes.op_type;
    try {
	let ret;
        switch (node.id) {
	case 'input_embed':
	    let curr_token_length = (phase === "regression") ? 1 : ctx.seq_length;
	    ret = op_eval(node.attributes.op_type, [batch, curr_token_length, dimension], {w_shape: formula_eval(node.attributes.formula, model_para), world_size: ctx.tp});
	    break;
	case 'layer_start':
	case 'layer_end':
	case 'v':    	    
	case 'shared_input':
	case 'routed_output':
	    ret = op_eval(node.attributes.op_type, pre_shape, {});
	    break;
	case 'q_nope':
	    ret = op_eval(node.attributes.op_type, pre_shape, {});
	    break
	case 'q_rope':
	case 'k_rope':
	    ret = op_eval(node.attributes.op_type, pre_shape, {w_shape: null, world_size: ctx.tp});
	    break;
	case 'kv_down_linear':
	case 'q_down_linear':
//	    ret = op_eval(op_type, pre_shape, {w_shape: formula_eval(node.attributes.formula, model_para), world_size: ctx.tp});
//	    break;
	case 'q_up_linear':
	case 'kv_up_linear':
	case 'q_linear':
	    ret = op_eval(op_type, pre_shape, {w_shape: formula_eval(node.attributes.formula, model_para), world_size: ctx.tp});
	    break;

        case 'k_linear':
        case 'v_linear':
            ret = op_eval(node.attributes.op_type, pre_shape, {w_shape: formula_eval(node.attributes.formula, model_para), world_size: ctx.tp, kv: true, input_seq: ctx.seq_length, output_seq: ctx.context_length});
            break;

	case 'q_cc':
	case 'k':
            ret = op_eval(node.attributes.op_type, pre_shape, {w_shape: formula_eval(node.attributes.formula, model_para)});
	    break;
	case 'qk_einsum':
	case 'qkv_einsum':
	    pre_shape[1][1] = (phase === "regression") ? (ctx.context_length + ctx.seq_length) : ctx.seq_length;
	    //pre_shape[1][2] = pre_shape[0][2]; //repeat kv dim to align with q dim
            ret = op_eval(node.attributes.op_type, pre_shape[0], { w_shape: pre_shape[1], world_size: ctx.tp, kv: true, input_seq: ctx.seq_length, output_seq: ctx.context_length});//[dimension, context_len]);	    
	    //ret = op_eval(node.attributes.op_type, pre_shape, {});
	    break;
	case 'q_split':    
	case 'kv_r_split':
	case 'kv_split':
	    console.log(node.attributes.outputs);
	    ret = op_eval(op_type, pre_shape, {w_shape: formula_eval(node.attributes.formula, model_para), outputs: node.attributes.outputs});
	    break;
	case 'kv_norm':
	    ret = op_eval(node.attributes.op_type, pre_shape, {w_shape: [dimension, vocab_size]});
	    break;
	case 'kv_up_linear_bak':
	    ret = op_eval(node.attributes.op_type, [batch, seq_len, model_para.kv_lora_rank], [model_para.kv_lora_rank, model_para.num_attention_heads*(model_para.qk_nope_head_dim + model_para.v_head_dim)]);
	    break;
	case 'moe_gate':
	    ret = op_eval(node.attributes.op_type, pre_shape, {w_shape: formula_eval(node.attributes.formula, model_para), world_size: ctx.tp});
	    break;
	case 'routed_up_proj':
	    ret = op_eval(node.attributes.op_type, [batch, seq_len, dimension], [model_para.hidden_size, model_para.moe_intermediate_size]);
	    break;
	case 'routed_feature_proj':
	    ret = op_eval(node.attributes.op_type, [batch, seq_len, dimension], [model_para.hidden_size, model_para.moe_intermediate_size]);
	    break;
	case 'routed_down_proj':
	    ret = op_eval(node.attributes.op_type, [batch, seq_len, dimension], [model_para.hidden_size, model_para.moe_intermediate_size]);
	    break;	    
	case 'shared_up_proj':
	    ret = op_eval(node.attributes.op_type, [batch, seq_len, dimension], [model_para.hidden_size, model_para.moe_intermediate_size]);
	    break;
	case 'shared_feature_proj':
	    ret = op_eval(node.attributes.op_type, [batch, seq_len, dimension], [model_para.hidden_size, model_para.moe_intermediate_size]);
	    break;
	case 'shared_down_proj':
	    ret = op_eval(node.attributes.op_type, [batch, seq_len, dimension], [model_para.hidden_size, model_para.moe_intermediate_size]);
	    break;	    
        case 'qk_matmul':
	    pre_shape[1][1] = (phase === "regression") ? (ctx.context_length + ctx.seq_length) : ctx.seq_length;
	    pre_shape[1][2] = pre_shape[0][2]; //repeat kv dim to align with q dim
            ret = op_eval(node.attributes.op_type, pre_shape[0], { w_shape: pre_shape[1], world_size: ctx.tp, kv: true, input_seq: ctx.seq_length, output_seq: ctx.context_length});//[dimension, context_len]);
            break;
        case 'qk_softmax':
            ret = op_eval(node.attributes.op_type, pre_shape, {world_size: ctx.tp});
            break;
        case 'sv_matmul':
	    pre_shape[1][1] = (phase === "regression") ? (ctx.context_length + ctx.seq_length) : ctx.seq_length;
	    pre_shape[1][2] = pre_shape[0][2]; //repeat kv dim to align with q dim
            ret = op_eval(node.attributes.op_type, pre_shape[0], { w_shape: pre_shape[1],/*[context_len, dimension]*/ world_size: ctx.tp, kv: true, input_seq: ctx.seq_length, output_seq:ctx.context_length});
            break;
        case 'attn_residual':
        case 'mlp_residual':
        case 'ffn_residual':	    
            ret = op_eval(node.attributes.op_type, pre_shape[0], {w_shape: pre_shape[1], world_size: ctx.tp});
            break;
	case 'output':    
	case 'o_linear':
	case 'shared_w1_linear':    
	case 'shared_w2_linear':    
	case 'shared_w3_linear':    
	case 'mlp_w1_linear':
	case 'mlp_w2_linear':
	case 'mlp_w3_linear':	    
	    ret = op_eval(op_type, pre_shape, {w_shape: formula_eval(node.attributes.formula, model_para), world_size: ctx.tp});
	    break;
	case 'routed_w1_linear':    
	case 'routed_w2_linear':    
	case 'routed_w3_linear':
	    ret = op_eval(op_type, pre_shape, {w_shape: formula_eval(node.attributes.formula, model_para)});
	    break;
	case 'mlp_act':
            ret = op_eval(node.attributes.op_type, pre_shape, {world_size: ctx.tp});
	    break;
	case 'shared_act':
	case 'routed_act':
	    console.log("mlp_act");
	    console.log(pre_shape);
            ret = op_eval(node.attributes.op_type, pre_shape, {});
	case 'routed_all_reduce':
            ret = op_eval(node.attributes.op_type, pre_shape, {});
	    break;
	case 'ffn_elmul':
	case 'mlp_elmul':
	    ret = op_eval(op_type, pre_shape[0], {w_shape: pre_shape[1], world_size: ctx.tp});
	    break;	    
	case 'shared_elmul':
	case 'routed_elmul':
	    ret = op_eval(op_type, pre_shape[0], {w_shape: pre_shape[1]});
	    break;
	case 'q_norm':
	    ret = op_eval(node.attributes.op_type, pre_shape, {w_shape: [dimension, vocab_size]});
	    break;
        case 'attn_norm':
        case 'mlp_norm':
	case 'ffn_norm':
	case 'last_layernorm':
            ret = op_eval(node.attributes.op_type, pre_shape, {w_shape: dimension});
            break;
        default:
            ret = op_eval(node.attributes.op_type, [batch, seq_len, dimension], [dimension, dimension]);
            break;
        }
	dev_op = JSON.parse(JSON.stringify(ret.dev));
	node_op = JSON.parse(JSON.stringify(ret.node));
    } catch (error) {
        console.error(`Error calculating metrics for ${node.id}:`, error);
    }
    updateBasicInfo(node_comp, node_op);
    updateBasicInfo(dev_comp, dev_op);
    calculateMemoryInfo(node_comp, node_op, ctx);
    calculateMemoryInfo(dev_comp, dev_op, ctx);
    calculateLatencyInfo(node_comp, dev_comp, ctx);
    calculateBoundInfo(node_comp);
    calculateBoundInfo(dev_comp);
    return {node_comp, dev_comp};
    function updateBasicInfo(dst, src) {
	const BASIC_INFO = [
	    'flops',
	    'in_shape',
	    'out_shape',
	    'w_shape',
	    'comm_para',
	    'comm_pattern'
	];
	const props = {};
	BASIC_INFO.forEach( key => {
	    props[key] = src[key];
	});
	Object.assign(dst, props);
    }
    function calculateMemoryInfo(dst, src,  ctx) {
	dst.mem.kv = src.kv_para * ctx.kv_width;
	dst.mem.weight = src.weight_para * ctx.w_width;
	dst.mem.acts = src.in_para * ctx.a_width;
	dst.mem.access = (src.ld_acts + src.st_acts) * ctx.a_width +
	    (src.ld_kv + src.st_kv) * ctx.kv_width +
	    (src.ld_weight) * ctx.w_width;
	dst.mem.peak = dst.mem.kv + dst.mem.weight; //+ dst.mem.acts;
	
    }
    function calculateBoundInfo(dst) {
        dst.ops2bytes = dst.mem.access > 0 ? dst.flops / dst.mem.access : 0;
        dst.bound = dst.latency.mem > dst.latency.comp ? 'memory' : 'compute';
    }	
    function calculateLatencyInfo(dst_node, dst_dev, ctx)
    {
	const __calc = (target, tp) => {
            const { mem, flops, comm_para } = target;
            const latency = {
		mem: (mem.access / (hw_spec.mem_bw * hw_spec.mem_bw_util * tp)) * 1e6,
		comp: (flops / (hw_spec.flops * tp)) * 1e6,
		comm: (comm_para * ctx.a_width / (hw_spec.scaleup_bw * hw_spec.scaleup_bw_util * tp)) * 1e6
            };
            latency.total = Math.max(latency.mem, latency.comp) + latency.comm;
            return latency;
	};
	const devLatency = __calc(dst_dev, 1);
	const nodeLatency = __calc(dst_node, ctx.tp);
	Object.assign(dst_dev.latency, devLatency);
	Object.assign(dst_node.latency, nodeLatency);
	dst_node.latency.total = dst_dev.latency.total + dst_node.latency.comm;

    }

}

module.exports = { model_eval };
