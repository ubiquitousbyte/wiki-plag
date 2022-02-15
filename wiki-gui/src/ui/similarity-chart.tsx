import {
    BarChart,
    CartesianGrid,
    Tooltip,
    XAxis,
    YAxis,
    Legend,
    Bar
} from "recharts";


export default function SimilarityChart(props: { data: { title: string, similarity: number }[] }) {
    return (
        <BarChart width={730} height={250} data={props.data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="title" />
            <YAxis />
            <Legend verticalAlign="top" height={36} />
            <Tooltip />
            <Bar name="Similarity to Input Document" dataKey="similarity" fill="#82ca9d" />
        </BarChart>
    )
}