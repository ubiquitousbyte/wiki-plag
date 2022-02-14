import React from "react";
import Paragraph from "../api/paragraph";


export default function ParagraphCard(props: { p: Paragraph }) {
    return (
        <div className="max-w-xl rounded oveflow-hidden border border-solid border-green-600 m-auto shadow-md">
            <div className="px-6 py-4">
                <div className="font-bold text-xl mb-2">{props.p.document}</div>
                <p className="text-gray-700 text-base">
                    {props.p.text}
                </p>
            </div>
        </div>
    )
}