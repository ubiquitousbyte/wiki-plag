import React from "react";
import Paragraph from "../api/paragraph";
import ParagraphList from "./paragraph-list";
import Spinner from "./spinner";

interface PlagConsoleProps {
    paragraphs: Paragraph[];
    error: string;
    loading: boolean;
}

export default function PlagConsole(props: PlagConsoleProps) {
    function render() {
        const { paragraphs, error, loading } = props;
        if (error.length > 0) {
            alert(error);
            return;
        }
        if (paragraphs.length > 0) {
            return <ParagraphList paragraphs={paragraphs} />
        } else {
            if (loading) {
                return <Spinner />;
            }
        }
    }
    return (
        <div className="h-full flex flex-col justify-center align-center">
            {render()}
        </div>
    )
}