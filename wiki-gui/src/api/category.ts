import client from "./config";
import APIError from "./error";
import { Category, Document } from "./entity";

let CategoryAPI = {
    getCategories: async function (start: number, offset: number): Promise<Category[] | APIError> {
        let params = { params: { start: start, offset: offset } }
        return client.get<Category[]>('/categories', params).
            then(
                (response) => response.data,
                (error) => APIError.fromResponse(error)
            )
    },
    getCategory: async function (id: string): Promise<Category | APIError> {
        return client.get<Category>(`/categories/${id}`)
            .then(
                (response) => response.data,
                (error) => APIError.fromResponse(error)
            )
    },
    getCategoryDocs: async function (id: string): Promise<Document[] | APIError> {
        return client.get<Document[]>(`/categories/${id}/documents`)
            .then(
                (response) => response.data,
                (error) => APIError.fromResponse(error)
            )
    },
    createCategory: async function (category: Category): Promise<string | APIError> {
        return client.post('/categories', category)
            .then(
                (response) => response.headers["Location"],
                (error) => APIError.fromResponse(error)
            )
    },
    deleteCategory: async function (id: string): Promise<void | APIError> {
        return client.delete(`/categories/${id}`)
            .then(
                (response) => { return; },
                (error) => APIError.fromResponse(error)
            )
    }
}

export default CategoryAPI;
