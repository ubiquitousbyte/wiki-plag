import React from "react";
import logo from "../htw-logo.png";
import github from "../github.png";

export default function Header() {
    return (
        <div className="flex items-center justify-between flex-wrap bg-green-500 p-6">
            <div className="flex items-center flex-shrink-0 text-white mr-6">
                <img src={logo} alt="HTW Logo" className="object-scale-down h-11 mr-6" />
            </div>
            <div className="w-full block flex-grow lg:flex lg:items-center lg:w-auto">
                <div className="text-sm lg:flex-grow">
                    <a className="flex gap-1 text-white font-bold" href="#">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 21h7a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v11m0 5l4.879-4.879m0 0a3 3 0 104.243-4.242 3 3 0 00-4.243 4.242z" />
                        </svg>
                        <span className="mt-1">Detector</span>
                    </a>
                </div>
                <a href="https://github.com/ubiquitousbyte/wiki-plag">
                    <img src={github} alt="Github" className="object-scale-down h-11 mr-6 text-white" />
                </a>
            </div>
        </div>
    );
}