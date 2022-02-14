import React, { useState } from "react";


export default function PlagForm(props: { onSubmit: (text: string) => Promise<void> | void }) {
    const [text, setText] = useState("");

    function onSubmit(e: React.SyntheticEvent) {
        e.preventDefault();
        props.onSubmit(text);
    }

    return (
        <form onSubmit={onSubmit}>
            <textarea
                className="
                form-control
                block
                w-full
                h-96
                px-3
                py-1.5
                text-base
                font-normal
                text-gray-700
                bg-white bg-clip-padding
                border border-solid border-gray-300
                rounded
                transition
                ease-in-out
                m-0
                mb-2
                focus:text-gray-700 focus:bg-white focus:border-green-600 focus:outline-none
                "
                rows={3}
                id="plagInput"
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Input suspicious document here">
            </textarea>
            <button className="
                 bg-green-500 
                 hover:bg-green-700 
                 text-white 
                 font-bold 
                 py-2 
                 px-4 
                 rounded
                "
                type="submit"
            >
                Detect
            </button>
        </form>
    )
}