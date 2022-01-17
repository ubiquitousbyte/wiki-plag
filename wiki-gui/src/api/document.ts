import client from "./config";
import APIError from "./error";
import { Document } from "./entity";


let DocumentAPI = {
    getDocument: async function (id: string): Promise<Document | APIError> {
        return client.get<Document>(`/document/${id}`)
            .then(
                (response) => response.data,
                (error) => APIError.fromResponse(error)
            )
    },
    createDocument: async function (document: Document): Promise<string | APIError> {
        return client.post('/document', document)
            .then(
                (response) => response.headers["Location"],
                (error) => APIError.fromResponse(error)
            )
    },
    updateDocument: async function (document: Document): Promise<void | APIError> {
        return client.put(`/document/${document.id}`, document)
            .then(
                (response) => { return; },
                (error) => APIError.fromResponse(error)

            )
    },
    deleteDocument: async function (id: string): Promise<void | APIError> {
        return client.delete(`/document/${id}`)
            .then(
                (response) => { return; },
                (error) => APIError.fromResponse(error)
            )
    }
}