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
        return new Promise(r => setTimeout(r, 1000))
            .then((r) => [
                new Paragraph("12", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),
                new Paragraph("13", "1", "This is some text", 3),
                new Paragraph("14", "1", "This is some text", 3),
                new Paragraph("15", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd  This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),
                new Paragraph("16", "1", "This is some text", 3),
                new Paragraph("18", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd  This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),

            ])
        /*  return client.post<Paragraph[]>('/plag', { text: text })
              .then((response) => response.data)
              .catch((error) => Promise.reject(APIError.fromResponse(error)));*/
        /*return Promise.resolve([
            new Paragraph("12", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),
            new Paragraph("13", "1", "This is some text", 3),
            new Paragraph("14", "1", "This is some text", 3),
            new Paragraph("15", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd  This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),
            new Paragraph("16", "1", "This is some text", 3),
            new Paragraph("18", "1", "This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd  This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd This is some more text blah blah blah blah asdas as as da dmagnaigna a s asd ", 3),

        ])*/
    }
}

export default PlagAPI;