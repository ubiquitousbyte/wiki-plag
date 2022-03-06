import axios from "axios";
import APIError from "./error";
import { Paragraph } from "./entity";

const client = axios.create({
    baseURL: "http://wikiplag.f4.htw-berlin.de:8081/api/v1",
    timeout: 1000 * 30,
    headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
})


class PlagCandidate {
    paragraph: Paragraph;
    similarity: number;

    constructor(paragraph: Paragraph, similarity: number) {
        this.paragraph = paragraph;
        this.similarity = similarity;
    }
}


const PlagAPI = {
    detect: async function (text: string): Promise<PlagCandidate[]> {
        return client.post<PlagCandidate[]>("/plag", { text: text })
            .then((response) => response.data)
            .catch((error) => {
                console.log(error);
                return Promise.reject(APIError.fromResponse(error));
            });
    }
}

export { PlagAPI, PlagCandidate };