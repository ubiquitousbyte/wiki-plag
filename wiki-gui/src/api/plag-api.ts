import axios from "axios";
import APIError from "./error";
import Paragraph from "./paragraph";

const client = axios.create({
    baseURL: "http://localhost:8081/api/v1",
    timeout: 1000 * 30,
    headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
})

const PlagAPI = {
    detect: async function (text: string): Promise<Paragraph[]> {
        /* return client.post<Paragraph[]>('/plag', { text: text })
             .then((response) => response.data)
             .catch((error) => Promise.reject(APIError.fromResponse(error)));*/
        return Promise.resolve([
            new Paragraph("1", "1", "This is some text", 3),
            new Paragraph("2", "1", "This is some text", 3),
            new Paragraph("3", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),
            new Paragraph("4", "1", "This is some text", 3),
            new Paragraph("5", "1", "This is some text", 3),
            new Paragraph("6", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),
            new Paragraph("7", "1", "This is some text", 3),
            new Paragraph("8", "1", "This is some text", 3),
            new Paragraph("9", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),
            new Paragraph("10", "1", "This is some text", 3),
            new Paragraph("11", "1", "This is some text", 3),
            new Paragraph("12", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),
            new Paragraph("13", "1", "This is some text", 3),
            new Paragraph("14", "1", "This is some text", 3),
            new Paragraph("15", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd  This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),
            new Paragraph("16", "1", "This is some text", 3),
        ])
    }
}

export default PlagAPI;