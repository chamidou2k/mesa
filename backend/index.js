const express = require('express');
const { model_list, load_model_graph, loadAvailModels, loadAvailHWSpecs,run_model_eval } = require('./model-parse.js');
const app = express();
const port = process.env.PORT || 9090;;
const cors = require('cors');
app.use(express.json());
app.use(cors());
app.get('/', (req, res) => {
    res.send('Model Evaluation and Smart Analytics (MESA) Tool');
});
app.get('/avail-profiles', (req, res) => {
    console.log('/avail-profiles:');
    res.json({
	message: {
	    models: loadAvailModels(),
	    hw_specs: loadAvailHWSpecs(),
	},
	status: 'success',
    });
});
app.get('/run_model_eval', (req, res) => {
    console.log('/run_model_eval:', req.query, "\n", model_list);
    console.dir(req.query.option)
    
    if (!req.query.option)
	return res.status(400).json({error: 'Missing "option" query parameter'});
    res.json({
	message: run_model_eval(JSON.parse(req.query.option)),
	status: 'success',
    });
});

app.listen(port, ()=> {
    console.log(`MESA Tool Backend running at localhost:${port}`);
});
