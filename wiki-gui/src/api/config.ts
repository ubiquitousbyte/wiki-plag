import axios from "axios";

const client = axios.create({
    baseURL: "http://localhost:8080/api/v1",
    timeout: 5000,
    headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
})

export default client; 