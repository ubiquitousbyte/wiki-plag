export default class APIError {
    detail: string;

    constructor(detail: string) {
        this.detail = detail;
    }

    static fromResponse(error: any): APIError {
        if (error.response) {
            return new APIError(error.response.data.detail);
        } else if (error.request) {
            return new APIError("Internal server error");
        } else {
            return new APIError(error.message);
        }
    }
}