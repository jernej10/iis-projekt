function convertDate(isoDate) {
    // Parse the ISO 8601 date
    const dateObject = new Date(isoDate);

    // Get day, month, and year
    const day = String(dateObject.getDate()).padStart(2, '0');
    const month = String(dateObject.getMonth() + 1).padStart(2, '0');
    const year = dateObject.getFullYear();

    // Format the date to "DD.MM.YYYY"
    const formattedDate = `${day}.${month}.${year}`;

    return formattedDate;
}

export default convertDate;
