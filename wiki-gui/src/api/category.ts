import client from "./config";
import APIError from "./error";
import { Category, Document } from "./entity";

let CategoryAPI = {
    getCategories: async function (start: number, offset: number): Promise<Category[] | APIError> {
        let params = { params: { start: start, offset: offset } }
        return client.get<Category[]>('/categories', params).
            then(
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
    getCategory: async function (id: string): Promise<Category | APIError> {
        return client.get<Category>(`/categories/${id}`)
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
    getCategoryDocs: async function (id: string): Promise<Document[] | APIError> {
        return client.get<Document[]>(`/categories/${id}/documents`)
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
    createCategory: async function (category: Category): Promise<string | APIError> {
        return client.post('/categories', category)
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
    deleteCategory: async function (id: string): Promise<void | APIError> {
        return client.delete(`/categories/${id}`)
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

export default CategoryAPI;
