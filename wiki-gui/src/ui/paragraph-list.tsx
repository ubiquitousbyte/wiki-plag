import React from "react";
import ParagraphCard from "./paragraph-card";
import Paragraph from "../api/paragraph";

export default function ParagraphList(props: { paragraphs: Paragraph[] }) {
    const paragraphs = props.paragraphs.map((p) =>
        <ParagraphCard key={p.id} p={p} />
    );

    return (
        <div className="grid grid-cols-2 gap-4">
            {paragraphs}
        </div>
    )
}