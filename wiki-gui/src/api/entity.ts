import internal from "stream";

class SparseDocument {
    id: string;
    title: string;

    constructor(id: string, title: string) {
        this.id = id;
        this.title = title;
    }
}

class Paragraph {
    id: string;
    document: SparseDocument;
    title: string;
    text: string;
    position: number;
    index: number;
    coordinates: [number, number];

    constructor(id: string, documentId: string, documentTitle: string, title: string,
        text: string, position: number, index: number,
        coordinates: [number, number]) {

        this.id = id;
        this.document = new SparseDocument(documentId, documentTitle);
        this.title = title;
        this.text = text;
        this.position = position;
        this.index = index;
        this.coordinates = coordinates;
    }
}


export { Paragraph, SparseDocument };
