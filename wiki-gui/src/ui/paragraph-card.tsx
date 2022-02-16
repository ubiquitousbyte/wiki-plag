import React from "react";
import { Paragraph } from "../api/entity";


export default function ParagraphCard(props: { paragraph: Paragraph }) {
    const { paragraph } = props;

    return (
        <div className="max-w-xl rounded oveflow-hidden border border-solid border-green-600 m-auto shadow-md">
            <div className="px-6 py-4">
                <div className="font-bold text-xl mb-2">{paragraph.document.title}</div>
                <div className="font-bold text-sm mb-2">{paragraph.title}</div>
                <p className="text-gray-700 text-base">
                    {paragraph.text}
                </p>
            </div>
        </div>
    )
}