import React, { useState } from "react";
import ParagraphCard from "./paragraph-card";
import Paragraph from "../api/paragraph";

export default function ParagraphList(props: { paragraphs: Paragraph[] }) {
    const [current, setCurrent] = useState(0);

    function onLeftClick() {
        let previous = current - 1;
        if (previous < 0) {
            setCurrent(props.paragraphs.length - 1);
        } else {
            setCurrent(previous);
        }
    }

    function onRightClick() {
        let next = current + 1;
        if (next >= props.paragraphs.length) {
            setCurrent(0);
        } else {
            setCurrent(next);
        }
    }

    return (
        <div className="grid grid-cols-6 gap-2 justify-items-center">
            <svg xmlns="http://www.w3.org/2000/svg"
                onClick={onLeftClick}
                className="h-6 w-6 m-auto fill-green-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor">
                <path strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            <div className="col-start-2 col-end-6 h-96 w-full overflow-auto flex align-center">
                <ParagraphCard p={props.paragraphs[current]} />
            </div>
            <svg xmlns="http://www.w3.org/2000/svg"
                onClick={onRightClick}
                className="h-6 w-6 m-auto fill-green-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor">
                <path strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
        </div>

    )
}