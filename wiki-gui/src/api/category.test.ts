import CategoryAPI from "./category";
import APIError from "./error";


describe("CategoryAPI", () => {
    it("retrieve categories from backend", () => {
        return CategoryAPI.getCategories(0, 1).then(data => {
            if (Array.isArray(data)) {
                expect(data.length).toBe(1);
            } else {
                fail("Unexpected response received when querying categories")
            }
        })
    })

    it("fail when requesting category that does not exist", () => {
        return CategoryAPI.getCategory("randomId").then(data => {
            expect(data instanceof APIError).toBeTruthy();
        })
    })
    it("retrieves category documents", () => {
        return CategoryAPI.getCategories(0, 1)
            .then(data => {
                if (Array.isArray(data) && data.length > 0) {
                    return CategoryAPI.getCategoryDocs(data[0].id)
                } else {
                    fail("Unexpected response received when querying categories");
                }
            })
            .then(documents => {
                if (Array.isArray(documents)) {
                    expect(documents.length > 0).toBeTruthy();
                } else {
                    fail("Unexpected response received when querying documents")
                }
            })
    })
})
