export default class Paragraph {
    id: string;
    document: string;
    text: string;
    position: number;

    constructor(id: string, document: string, text: string, position: number) {
        this.id = id;
        this.document = document;
        this.text = text;
        this.position = position;
    }
}
