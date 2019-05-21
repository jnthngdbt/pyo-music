const express = require('express');
const app = express();

app.get('/', (req,res) => {
    console.log('GET');
    // res.set({ // set sets header
    //     'what': 'cool',
    //     'yes': 'ok',
    // });
    res.send({ // send sets body
        'what': 'cool',
        'yes': 'ok',
    });

    res.end(); // stops the request
});

app.post('/', (req,res) => {
    console.log('POST');
    console.log(req.body);

    res.send({ // send sets body
        'res': 'cool',
        'notreq': 'ok',
    });

    res.end();
});

app.listen(3000, () => console.log('Running at 3000'));