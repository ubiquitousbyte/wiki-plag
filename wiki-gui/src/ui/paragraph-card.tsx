import React from "react";
import Paragraph from "../api/paragraph";


export default function ParagraphCard(props: { p: Paragraph }) {
    console.log(props.p.document);
    return (
        <div className="max-w-sm rounded oveflow-hidden shadow-lg">
            <div className="px-6 py-4">
                <div className="font-bold text-xl mb-2">{props.p.document}</div>
                <p className="text-gray-700 text-base">
                    {props.p.text}
                </p>
            </div>
        </div>
    )
}