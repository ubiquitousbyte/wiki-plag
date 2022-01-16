import client from "./config";
import APIError from "./error";
import { Document } from "./entity";


let DocumentAPI = {
    getDocument: async function (id: string): Promise<Document | APIError> {
        return client.get<Document>(`/document/${id}`)
            .then(
                (response) => response.data,
                (error) => {
                    if (error.response) {
                        return new APIError(error.response.data.detail);
                    } else if (error.request) {
                        return new APIError("Fatal error");
                    } else {
                        return new APIError(error.message);
                    }
                }
            )
    },
    createDocument: async function (document: Document): Promise<string | APIError> {
        return client.post('/document', document)
            .then(
                (response) => response.headers["Location"],
                (error) => {
                    if (error.response) {
                        return new APIError(error.response.data.detail);
                    } else if (error.request) {
                        return new APIError("Fatal error");
                    } else {
                        return new APIError(error.message);
                    }
                }
            )
    },
    updateDocument: async function (document: Document): Promise<void | APIError> {
        return client.put(`/document/${document.id}`, document)
            .then(
                (response) => { return; },
                (error) => {
                    if (error.response) {
                        return new APIError(error.response.data.detail);
                    } else if (error.request) {
                        return new APIError("Fatal error");
                    } else {
                        return new APIError(error.message);
                    }
                }
            )
    },
    deleteDocument: async function (id: string): Promise<void | APIError> {
        return client.delete(`/document/${id}`)
            .then(
                (response) => { return; },
                (error) => {
                    if (error.response) {
                        return new APIError(error.response.data.detail);
                    } else if (error.request) {
                        return new APIError("Fatal error");
                    } else {
                        return new APIError(error.message);
                    }
                }
            )
    }
}