import React, { Component } from 'react';
import { LineChart } from 'react-easy-chart';

export default class Graph extends Component {
    render() {
        return (
            <LineChart
                margin={{top: 30, right: 30, bottom: 30, left: 30}}
                width={250}
                height={250}
                data={[
                [
                    { x: 1, y: 20 },
                    { x: 2, y: 10 },
                    { x: 3, y: 25 }
                ], [
                    { x: 1, y: 10 },
                    { x: 2, y: 12 },
                    { x: 3, y: 4 }
                ]
                ]}
            />
        );
    }
}