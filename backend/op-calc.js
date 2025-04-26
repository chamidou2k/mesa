function op_eval(op_type, in_shape, {w_shape = null,  world_size = 1, kv = false, input_seq = 0, output_seq = 0, outputs = null}) {
    let valid = false;
    let flops = 0, flops_n = 0;
    let base = 1;
    let weight_para = 0, weight_para_n = 0;
    let in_para = 0, in_para_n = 0;
    let kv_para = 0, kv_para_n = 0;
    let is_kv_store = false, is_kv_load = false;
    let out_para = 0, out_para_n = 0;
    let out_shape = {}, out_shape_n = {};
    let w_shape_n = 0, in_shape_n = 0;
    let coll_para = 0;
    let coll_pattern = 'NA';

    if (!w_shape || w_shape.length === 0) {
        w_shape = 0;
    }    
    console.log(w_shape);
    console.log(`${op_type} in: ${JSON.stringify(in_shape)} | ${JSON.stringify(w_shape)}`);
    switch (op_type) {
    case 'Dummy':
        valid = true;
        in_para = 0;
        out_para = 0;
        weight_para = 0;
        out_shape_n = [...in_shape];
        in_shape_n = [...in_shape];
        out_shape = [...out_shape_n];
        break;
    case 'Router':
        in_para = 0;
        out_para = 0;
        weight_para = 0;
        out_shape_n = [...in_shape];
        in_shape_n = [...in_shape];
        out_shape = [...out_shape_n];
        break;
    case 'all_reduce':
        in_para = 0; 
        out_para = 0;
        weight_para = 0;
        out_shape_n = [...in_shape];
        in_shape_n = [...in_shape];
        out_shape = [...out_shape_n];
        if (world_size > 1) {
            coll_para = (world_size - 1) / world_size * out_para * world_size;
            coll_pattern = 'all-reduce';
        }
        break;
    case 'RoPE':
        {
            let round = in_shape.reduce((acc, item) => acc * item, 1);
            flops = 6 * round / 2;
            in_para = round;
            out_shape = [...in_shape];
            out_para = in_para;
            flops_n = flops * world_size;
            in_shape_n = [...in_shape];
            out_shape_n = [...out_shape];
            in_shape_n[in_shape_n.length -1] *= world_size;
            out_shape_n[out_shape_n.length -1] *= world_size;
            in_para_n = in_para * world_size;
            out_para_n = out_para * world_size;
        }
        break;
    case 'concat':
        {
            let round = in_shape.reduce((acc, item) => Array.isArray(item) ? acc + item.reduce((a, b) => a * b, 1) : acc * item, 1);
            flops = 0;
            in_para = round;
            out_shape = [...in_shape[0]];
            out_shape[out_shape.length - 1] += in_shape[1][in_shape[1].length - 1];
            out_para = out_shape.reduce((acc, item) => acc * item, 1);
            flops_n = flops * world_size;
            in_shape_n = [...in_shape];
            out_shape_n = [...out_shape];
            in_shape_n[in_shape_n.length -1] *= world_size;
            out_shape_n[out_shape_n.length -1] *= world_size;
            in_para_n = in_para * world_size;
            out_para_n = out_para * world_size;
        }
        break;
    case 'split':
        {
            let round = in_shape.reduce((acc, item) => acc * item, 1);
            flops = 0;
            in_para = round;
            outputs.forEach((output, i) => {
                out_shape[output] = [...in_shape];
                out_shape[output][out_shape[output].length - 1] = w_shape[i];
            });
            out_para = outputs.reduce((total, key) => {
                const shape = out_shape[key];
                if (!shape || !Array.isArray(shape)) return total;
                return total + shape.reduce((acc, dim) => acc * dim, 1);
            }, 0);
            flops_n = flops * world_size;
            in_shape_n = [...in_shape];
            outputs.forEach((output) => {
                out_shape_n[output] = [...out_shape[output]];
            });
            in_para_n = in_para * world_size;
            out_para_n = out_para * world_size;
            break;
        }
    case 'ParallelEmbedding':
        w_shape_n = [...w_shape];
        in_shape = [...in_shape];
        w_shape[0] = w_shape[0] / world_size;
        out_shape = [...in_shape];
        weight_para = w_shape.reduce((acc, item) => acc * item, 1);
        in_para = in_shape.reduce((acc, item) => acc * item, 1);
        out_para = in_para;
        in_shape_n = in_shape;
        in_para_n = in_para * world_size;
        out_shape_n = in_shape_n;
        out_para_n = out_para * world_size;
        weight_para_n = weight_para * world_size;
        if (world_size > 1) {
            coll_para = (world_size - 1) / world_size * out_para * world_size;
            coll_pattern = 'all-reduce';
        }
        break;
    case 'ColumnParallelLinear':
        {
            w_shape_n = [...w_shape];
            w_shape[w_shape.length - 1] /= world_size;
            let round = in_shape.slice(0, -1).reduce((acc, item) => acc * item, 1) * w_shape.slice(1).reduce((acc, item) => acc * item, 1);
            flops = round * w_shape[0] + round * (w_shape[0] - 1) + round;
            weight_para = w_shape.reduce((acc, item) => acc * item, 1);
            in_para = in_shape.reduce((acc, item) => acc * item, 1);
            const part_a = in_shape.slice(0, -1);
            let part_b = w_shape.slice(1);
            out_shape = part_a.concat(part_b);
            out_para = out_shape.reduce((acc, item) => acc * item, 1);
            flops_n = flops * world_size;
            in_para_n = in_para;
            out_para_n = out_para;
            in_shape_n = [...in_shape];
            out_shape_n = [...out_shape];
            out_shape_n[out_shape_n.length -1] *= world_size;
            weight_para_n = weight_para * world_size;
            if (kv) {
                kv_para = out_para;
                kv_para_n = out_para_n;
                is_kv_store = true;
                out_para = 0;
                out_para_n = 0;
            }
        }
        break;
    case 'RowParallelLinear':
        {
            w_shape_n = [...w_shape];
            w_shape[0] /= world_size;
            let round = in_shape.slice(0, -1).reduce((acc, item) => acc * item, 1) * w_shape.slice(0, -1).reduce((acc, item) => acc * item, 1);
            flops = round * w_shape[w_shape.length - 1] + round * (w_shape[w_shape.length - 1] - 1) + round;
            weight_para = w_shape.reduce((acc, item) => acc * item, 1);
            in_para = in_shape.reduce((acc, item) => acc * item, 1);
            const part_a = in_shape.slice(0, -1);
            let part_b = w_shape.slice(1);
            out_shape = part_a.concat(part_b);
            out_para = out_shape.reduce((acc, item) => acc * item, 1);
            flops_n = flops * world_size;
            in_para_n = in_para * world_size;
            out_para_n = out_para * world_size;
            in_shape_n = [...in_shape];
            in_shape_n[in_shape_n.length -1] *= world_size;
            out_shape_n = [...out_shape];
            weight_para_n = weight_para * world_size;
            if (world_size > 1) {
                coll_para = (world_size - 1) / world_size * out_para * world_size;
                coll_pattern = 'all-reduce';
            }
        }
        break;
    case 'Linear':
    case 'Gemm':
        {
            w_shape_n = [...w_shape];
            let round = in_shape.slice(0, -1).reduce((acc, item) => acc * item, 1) * w_shape.slice(1).reduce((acc, item) => acc * item, 1);
            flops = round * w_shape[0] + round * (w_shape[0] - 1) + round;
            weight_para = w_shape.reduce((acc, item) => acc * item, 1);
            in_para = in_shape.reduce((acc, item) => acc * item, 1);
            const part_a = in_shape.slice(0, -1);
            let part_b = w_shape.slice(1);
            out_shape = part_a.concat(part_b);
            out_para = out_shape.reduce((acc, item) => acc * item, 1);
            flops_n = flops;
            in_para_n = in_para;
            out_para_n = out_para;
            in_shape_n = [...in_shape];
            out_shape_n = [...out_shape];
            weight_para_n = weight_para;
            break;
        }
    case 'einsum':  	
    case 'AttnScore':
        {
            valid = true;
            let round = in_shape.slice(0, -1).reduce((acc, item) => acc * item, 1) * w_shape[1];
            flops = round * w_shape[2] + round * (w_shape[2] - 1);
            weight_para = w_shape.reduce((acc, item) => acc * item, 1);
            in_para = in_shape.reduce((acc, item) => acc * item, 1);
            out_shape = [...in_shape];
            out_para = out_shape.reduce((acc, item) => acc * item, 1);
            flops_n = flops * world_size;
            in_para_n = in_para * world_size;
            out_para_n = out_para * world_size;
            in_shape_n = [...in_shape];
            out_shape_n = [...out_shape];
            in_shape_n[in_shape_n.length -1] *= world_size;
            out_shape_n[out_shape_n.length -1] *= world_size;
            weight_para_n = weight_para * world_size;
            if (kv) {
                kv_para = weight_para;
                kv_para_n = weight_para_n;
                is_kv_load = true;
                weight_para = 0;
                weight_para_n = 0;
            }
        }
        break;
    case 'Matmul':
        if (in_shape[in_shape.length -1] == w_shape[0]) {
            valid = true;
            let round = in_shape.slice(0, -1).reduce((acc, item) => acc * item, 1) * w_shape.slice(1).reduce((acc, item) => acc * item, 1);
            flops = round * w_shape[0] + round * (w_shape[0] - 1);
            weight_para = w_shape.reduce((acc, item) => acc * item, 1);
            in_para = in_shape.reduce((acc, item) => acc * item, 1);
            const part_a = in_shape.slice(0, -1);
            let part_b = w_shape.slice(1);
            out_shape = part_a.concat(part_b);
            out_para = out_shape.reduce((acc, item) => acc * item, 1);
        }
        break;
    case 'Elmul':
        {
            valid = true;
            let round = in_shape.slice(0, -1).reduce((acc, item) => acc * item, 1);
            flops = round;
            in_para = in_shape.reduce((acc, item) => acc * item, 1) * 2;
            out_shape = [...in_shape];
            out_para = out_shape.reduce((acc, item) => acc * item, 1);
            flops_n = flops * world_size;
            in_para_n = in_para * world_size;
            out_para_n = out_para * world_size;
            in_shape_n = [...in_shape];
            out_shape_n = [...out_shape];
            in_shape_n[in_shape_n.length -1] *= world_size;
            out_shape_n[out_shape_n.length -1] *= world_size;
        }
        break;
    case "RMSNorm":
        valid = true;
        base = in_shape.reduce((acc, item) => acc * item, 1);
        flops = 5 * base;
        in_para = base;
        out_shape = [...in_shape];
        out_para = base;
        weight_para = in_shape[in_shape.length - 1];
        flops_n = flops;
        in_shape_n = [...in_shape];
        in_para_n = in_para;
        out_shape_n = [...in_shape_n];
        out_para_n = out_para;
        weight_para_n = weight_para;
        break;
    case "LayerNorm":
        valid = true;
        base = in_shape.reduce((acc, item) => acc * item, 1);
        flops = 7 * base;
        in_para = base;
        out_shape = [...in_shape];
        out_para = base;
        weight_para = in_shape[in_shape.length - 1];
        break;
    case "BatchNorm":
        valid = true;
        base = in_shape.reduce((acc, item) => acc * item, 1);
        flops = 7 * base;
        in_para = base;
        out_shape = [...in_shape];
        out_para = base;
        weight_para = 2 * in_shape[in_shape.length - 1];
    case "Add":
        valid = true;
        base = in_shape.reduce((acc, item) => acc * item, 1);
        flops = base;
        in_para = base;
        out_shape = [...in_shape];
        out_para = base;
        flops_n = flops;
        in_shape_n = [...in_shape];
        in_para_n = in_para;
        out_shape_n = [...in_shape_n];
        out_para_n = out_para;
        weight_para_n = weight_para;
        break;
    case "Softmax":
        valid = true;
        base = in_shape.reduce((acc, item) => acc * item, 1);
        flops = 5 * base;
        in_para = base;
        out_para = base;
        out_shape = [...in_shape];
        flops_n = flops * world_size;
        in_para_n = in_para * world_size;
        out_para_n = out_para * world_size;
        in_shape_n = [...in_shape];
        out_shape_n = [...out_shape];
        in_shape_n[in_shape_n.length -1] *= world_size;
        out_shape_n[out_shape_n.length -1] *= world_size;
        break;
    case "Relu":
        valid = true;
        base = in_shape.reduce((acc, item) => acc * item, 1);
        flops = 2 * base;
        in_para = base * 2;
        out_para = base;
        break;
    case "Silu":
        valid = true;
        base = in_shape.reduce((acc, item) => acc * item, 1);
        flops = 2 * base;
        in_para = base * 2;
        out_para = base;
        out_shape = [...in_shape];
        flops_n = flops * world_size;
        in_para_n = in_para * world_size;
        out_para_n = out_para * world_size;
        in_shape_n = [...in_shape];
        out_shape_n = [...out_shape];
        in_shape_n[in_shape_n.length -1] *= world_size;
        out_shape_n[out_shape_n.length -1] *= world_size;
        break;
    case "moe_router":
        valid = true;
        break;
    default:
        break;
    }
    const dev_op  = {
        in_shape: in_shape,
        w_shape: w_shape,
        out_shape: out_shape,
        in_para: in_para,
        kv_para: (kv ? kv_para : 0),
        weight_para: weight_para,
        out_para: (!kv ? out_para :  0),
        flops: flops,
        comm_pattern: coll_pattern,
        comm_para: 0,
        ld_acts: in_para, 
        ld_kv: (is_kv_load ? kv_para : 0), 
        ld_weight: weight_para, 
        st_acts: out_para, 
        st_kv: (is_kv_store ? kv_para : 0), 
    };
    const node_op  = {
        in_shape: in_shape_n,
        w_shape: w_shape_n,
        out_shape: out_shape_n,
        in_para: in_para_n,
        weight_para: weight_para_n,
        kv_para: (kv ? kv_para_n : 0), 
        out_para: (!kv ? out_para_n :  0),
        flops: flops_n,
        comm_pattern: coll_pattern,
        comm_para: coll_para,
        ld_acts: in_para_n, 
        ld_kv: (is_kv_load ? kv_para_n : 0), 
        ld_weight: weight_para_n, 
        st_acts: out_para_n, 
        st_kv: (is_kv_store ? kv_para_n : 0), 
    };

    //console.log(`${op_type} block_op: ${JSON.stringify(dev_op)}`);
    //console.log(`${op_type} dev_op: ${JSON.stringify(node_op)}`);
    return { node: node_op, dev: dev_op };
}
module.exports = {
    op_eval
}
