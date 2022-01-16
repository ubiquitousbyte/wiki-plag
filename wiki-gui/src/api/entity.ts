class Category {
    id: string;
    source: string;
    name: string;
    description: string;

    constructor(id: string, source: string, name: string, description: string) {
        this.id = id;
        this.source = source;
        this.name = name;
        this.description = description;
    }
}

class Document {
    id: string;
    title: string;
    source: string;
    categories: string[];

    constructor(id: string, title: string, source: string, categories: string[]) {
        this.id = id;
        this.title = title;
        this.source = source;
        this.categories = categories;
    }
}

export { Category, Document };